"""Prepare isolated, calibrated COLMAP datasets for aerial/ground training."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from colmap_io import read_model, write_model
from apply_sky_mask import process_one
from selectpairs import select_pairs

ROOT = Path(__file__).resolve().parents[1]


def add_arguments(parser):
    parser.add_argument('--skyanchor_run', type=Path, help='Completed run_skyanchor output directory')
    parser.add_argument('--prepare_only', action='store_true', help='Prepare images, sky masks, models and pair files, then exit')
    parser.add_argument('--skip_sky_removal', action='store_true', help='Keep source alpha; do not run SkyRemoval on RGB ground images')
    parser.add_argument('--mask_aerial', action='store_true', help='Also remove sky from aerial images')
    parser.add_argument('--sky_model', type=Path, default=ROOT/'splatting/submodules/SkyRemoval/model/model.onnx')
    parser.add_argument('--pair_workers', type=int, default=8)


def source_image(folder, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe COLMAP image name: {name}')
    exact = folder / relative
    if exact.is_file():
        return exact
    png = exact.with_suffix('.png')
    if png.is_file():
        return png
    raise FileNotFoundError(exact)


def input_stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def prepare_dataset(model, images_dir, destination, mask=False, sky_model=None,
                    scale=1.0, pair_file='', dense_name='dense_v2.ply', workers=8, downsampling=None):
    """Preserve image/point IDs and tracks; copy models and scale both t and XYZ."""
    model, images_dir, destination = map(Path, (model, images_dir, destination))
    result = read_model(str(model))
    if result is None:
        raise ValueError(f'No COLMAP model: {model}')
    cameras, images, points = result
    if not images or not points:
        raise ValueError(f'Empty registered cameras or points in {model}')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid scene scale')
    for camera in cameras.values():
        if camera.model not in ('PINHOLE', 'SIMPLE_PINHOLE'):
            raise ValueError(f'Undistorted PINHOLE images required: {camera.model}')
    sources = {i: source_image(images_dir, im.name) for i,im in images.items()}
    names = {i: str(Path(im.name).with_suffix('.png')) for i,im in images.items()}
    if len(set(names.values())) != len(names):
        raise ValueError('Image names collide after PNG conversion')
    stamp = {'version': 2, 'downsampling': downsampling, 'model': [input_stamp(p) for p in sorted(model.glob('*')) if p.suffix in ('.txt','.bin')],
             'images': [input_stamp(p) for p in sources.values()], 'mask': mask, 'scale': scale,
             'sky_model': input_stamp(sky_model) if mask else None,
             'pair_file': input_stamp(pair_file) if pair_file else None, 'dense_name': dense_name}
    fingerprint = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    manifest = destination/'prepared.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['fingerprint'] != fingerprint:
            raise ValueError(f'Inputs changed; choose a new training output directory: {destination}')
        for name in names.values():
            if not (destination/'images'/name).is_file():
                raise FileNotFoundError(destination/'images'/name)
        if not (destination/'pair.txt').exists():
            raise FileNotFoundError(destination/'pair.txt')
        print(f'[Prepare] Reusing verified inputs: {destination}', flush=True)
        return destination
    sparse = destination/'sparse/0'
    sparse.mkdir(parents=True, exist_ok=True)
    # A partial preparation may be retried, but no generated dense point cloud is reused.
    if (sparse/dense_name).exists():
        raise ValueError(f'Unverified dense cache in {sparse}; use a new output directory')
    segmenter = None
    for index,(iid,im) in enumerate(images.items(),1):
        camera = cameras[im.camera_id]
        out = destination/'images'/names[iid]
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(sources[iid]) as source:
            rgba = 'A' in source.getbands()
            array = np.asarray(source.convert('RGBA' if rgba else 'RGB'))
        if (array.shape[1],array.shape[0]) != (camera.width,camera.height):
            import cv2
            if downsampling is not None:
                # Match anchorRecon's pixel-center sampling, including odd image heights.
                array = cv2.resize(array,(0,0),fx=1/downsampling,fy=1/downsampling,interpolation=cv2.INTER_AREA)
            else:
                array = cv2.resize(array,(camera.width,camera.height),interpolation=cv2.INTER_AREA)
            if (array.shape[1],array.shape[0]) != (camera.width,camera.height):
                raise ValueError(f'Downsampling disagrees with exported camera: {im.name}')
        Image.fromarray(array).save(out, compress_level=1)
        if mask and not rgba:
            if segmenter is None:
                from submodules.SkyRemoval.skyfilter import SkyFilter
                segmenter = SkyFilter(str(sky_model))
            mask_array = segmenter.get_mask(array[:,:,:3].astype(np.float32)/255.)
            mask_file = destination/'masks'/names[iid]
            mask_file.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(mask_array).save(mask_file)
            process_one(out,mask_file,out,128)
        images[iid] = im._replace(name=names[iid], tvec=im.tvec*scale)
        if index == 1 or index % 50 == 0 or index == len(images):
            print(f'[Prepare] {destination.name}: {index}/{len(images)} images', flush=True)
    points = {pid: p._replace(xyz=p.xyz*scale) for pid,p in points.items()}
    write_model(cameras,images,points,str(sparse),ext='.txt')
    # Reuse explicitly named dense input only when coordinate scaling is unnecessary.
    dense = Path(dense_name)
    dense_source = dense if dense.is_absolute() else (model/dense if len(dense.parts)==1 else model.parent.parent/dense)
    if dense_source.is_file() and dense_source.name != 'points3D.ply':
        if scale != 1:
            raise ValueError('Cannot reuse unscaled dense PLY with scene_scale != 1; regenerate dense initialization')
        shutil.copy2(dense_source,sparse/'dense_v2.ply')
    if pair_file:
        # Pair files reference IDs, so PNG renaming does not alter their semantics.
        lines = Path(pair_file).read_text().splitlines()
        if not lines or int(lines[0]) != len(images) or len(lines) != 1+2*len(images):
            raise ValueError(f'Invalid pair file coverage: {pair_file}')
        refs=set()
        for a,b in zip(lines[1::2],lines[2::2]):
            ref=int(a); fields=b.split(); count=int(fields[0])
            if ref not in images or ref in refs or len(fields)!=1+2*count:
                raise ValueError(f'Invalid pair reference: {a}')
            refs.add(ref)
            if any(int(fields[1+2*k]) not in images or int(fields[1+2*k])==ref for k in range(count)):
                raise ValueError(f'Invalid pair neighbors: {b}')
        shutil.copy2(pair_file,destination/'pair.txt')
    else:
        select_pairs(sparse,destination/'pair.txt',workers=workers)
    manifest.write_text(json.dumps({'fingerprint':fingerprint,'source':stamp,'images':len(images),'points':len(points)},indent=2)+'\n')
    return destination


def prepare_training_inputs(args):
    if args.skyanchor_run:
        run = args.skyanchor_run.resolve()
        config = json.loads((run/'run.json').read_text())
        aerial_model, aerial_images = Path(config['aerial_model']), Path(config['aerial_images'])
        ground_model, ground_images = run/'ground/final_ba_colmap',Path(config['ground_images'])
        scale = float(config['scene_scale'])
        downsampling = float(config.get('downsampling', 1.))
    else:
        if not args.source_path:
            raise ValueError('Provide -s or --skyanchor_run')
        aerial_model = Path(args.source_path)/'sparse/0'
        aerial_images = Path(args.source_path)/args.images
        ground_model = Path(args.ground_source_path)/'sparse/0' if args.ground_source_path else None
        ground_images = Path(args.ground_source_path)/args.ground_images if ground_model else None
        scale = 1.
        downsampling = None
    root = Path(args.model_path).resolve()/'prepared'
    aerial = prepare_dataset(aerial_model,aerial_images,root/'aerial',args.mask_aerial and not args.skip_sky_removal,
                             args.sky_model,scale,args.pair_file,args.dense_init_ply_name,args.pair_workers)
    args.source_path, args.images, args.pair_file = str(aerial),'images',str(aerial/'pair.txt')
    if ground_model:
        ground = prepare_dataset(ground_model,ground_images,root/'ground',not args.skip_sky_removal,
                                  args.sky_model,1.,args.ground_pair_file,
                                  args.ground_dense_init_ply_name or args.dense_init_ply_name,args.pair_workers,downsampling)
        args.ground_source_path,args.ground_images,args.ground_pair_file = str(ground),'images',str(ground/'pair.txt')
    if args.dense_init_ply_name != 'points3D.ply':
        args.dense_init_ply_name = 'dense_v2.ply'
    if args.ground_dense_init_ply_name and args.ground_dense_init_ply_name != 'points3D.ply':
        args.ground_dense_init_ply_name = 'dense_v2.ply'
    return args
