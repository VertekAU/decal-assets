"""Vertek decal customer preview: decal art inside the chrome tap frame over a bar backdrop, with a
tiled watermark. Static art -> "<out_base>.jpg" still; animated art -> "<out_base>.mp4".
Usage: python build_preview.py <decal.mp4 or image> "<out dir>/Preview - <name>" [frame.png]
The frame defaults to FRAME_URL (downloaded each run); pass a local path to override.
Prints the path of the file it wrote."""
import os, sys, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage as nd

W, H = 1920, 1080
S = 1.12          # frame scale (keeps the Vertek badge in shot)
FRAME_TOP = 24
FRAME_URL = 'https://raw.githubusercontent.com/VertekAU/decal-assets/main/decal-frame-transparent.png'


def fetch_frame(src, tmp):
    """Return a local path for the frame: download if src is a URL, else use it as-is."""
    if not src.startswith('http'):
        return src
    dest = os.path.join(tmp, 'decal-frame.png')
    try:   # curl ships with Windows 10+, macOS and Linux, and honours proxy/CA settings
        subprocess.run(['curl', '-fsSL', '-o', dest, src], check=True)
    except (OSError, subprocess.CalledProcessError):
        import urllib.request
        urllib.request.urlretrieve(src, dest)
    return dest


def font(size):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              'C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/segoeuib.ttf',
              '/System/Library/Fonts/Supplemental/Arial Bold.ttf']:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def bar_background(rng):
    y = np.linspace(0, 1, H)[:, None]; x = np.linspace(0, 1, W)[None, :]
    base = np.zeros((H, W, 3)); base[:] = [28, 16, 10]
    base += (np.exp(-((y - 0.45) ** 2) / 0.08) * np.ones_like(x))[..., None] * [70, 38, 14]
    bg = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert('RGBA')

    back = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(back)
    glass = [(150, 85, 20), (90, 120, 40), (200, 170, 110), (120, 40, 20),
             (60, 90, 60), (210, 140, 40), (170, 180, 170), (80, 30, 40)]
    for sy in [250, 520, 790]:
        d.rectangle([0, sy, W, sy + 14], fill=(60, 35, 20, 255))
        d.rectangle([0, sy - 3, W, sy + 1], fill=(255, 190, 110, 200))
        xx = -20
        while xx < W:
            bw = int(rng.integers(34, 62)); bh = int(rng.integers(120, 210))
            c = glass[rng.integers(len(glass))]; top = sy - bh; nk = bw // 3
            col = tuple(int(v * rng.uniform(.7, 1.1)) for v in c) + (255,)
            d.rounded_rectangle([xx, top + bh * 0.35, xx + bw, sy], radius=bw // 3, fill=col)
            d.rectangle([xx + bw // 2 - nk // 2, top, xx + bw // 2 + nk // 2, top + bh * 0.4], fill=col)
            d.rectangle([xx + bw // 2 - nk // 2 - 2, top, xx + bw // 2 + nk // 2 + 2, top + 10], fill=(40, 30, 25, 255))
            d.rectangle([xx + 6, top + bh * 0.4, xx + 11, sy - 8],
                        fill=tuple(min(255, v + 90) for v in col[:3]) + (180,))
            xx += bw + int(rng.integers(4, 22))
    bg = Image.alpha_composite(bg, back.filter(ImageFilter.GaussianBlur(14)))

    bok = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(bok)
    for _ in range(70):
        cx, cy = rng.integers(0, W), rng.integers(0, int(H * 0.8)); r = int(rng.integers(14, 46))
        c = [(255, 200, 120), (255, 170, 80), (255, 230, 180), (200, 120, 60)][rng.integers(4)]
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c + (int(rng.integers(30, 90)),))
    bg = Image.alpha_composite(bg, bok.filter(ImageFilter.GaussianBlur(3)))

    ct = 900
    yy = np.linspace(0, 1, H - ct)[:, None]
    cnt = np.zeros((H - ct, W, 3)); cnt[:] = [62, 34, 18]; cnt *= (1 - 0.5 * yy)[..., None]
    cnt += (nd.gaussian_filter(rng.normal(0, 1, (H - ct, W)), (1, 40)) * 25)[..., None] * [1, .6, .35]
    cnt += (np.exp(-((yy - 0.08) ** 2) / 0.002) * np.ones((1, W)))[..., None] * [90, 60, 35]
    bg.paste(Image.fromarray(np.clip(cnt, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), (0, ct))
    ImageDraw.Draw(bg).rectangle([0, ct - 4, W, ct + 2], fill=(150, 100, 60, 255))

    v = 1 - 0.55 * np.clip(((x - 0.5) ** 2 * 1.6 + (y - 0.5) ** 2 * 1.2) * 2.2, 0, 1)
    return Image.fromarray((np.array(bg.convert('RGB')).astype(float) * v[..., None]).astype(np.uint8))


def frame_layer(frame_path):
    fr = Image.open(frame_path).convert('RGBA')
    clear = np.array(fr)[..., 3] < 128
    lab, n = nd.label(clear)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    sizes = nd.sum(clear, lab, range(1, n + 1))
    inner = [(sizes[i - 1], i) for i in range(1, n + 1) if i not in border]
    hole = lab == max(inner)[1]                      # largest enclosed transparent region = screen hole
    ys, xs = np.nonzero(hole)
    hcx, hcy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    r = ((xs.max() - xs.min()) + (ys.max() - ys.min())) / 4

    fr = fr.resize((round(fr.width * S), round(fr.height * S)), Image.LANCZOS)
    hcx, hcy, r = hcx * S, hcy * S, r * S
    fx, fy = round(W / 2 - hcx), FRAME_TOP
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    P = 60
    sh = Image.new('RGBA', (fr.width + 2 * P, fr.height + 2 * P), (0, 0, 0, 0))
    sa = Image.new('L', sh.size, 0); sa.paste(fr.getchannel('A').point(lambda p: int(p * 0.55)), (P, P)); sh.putalpha(sa)
    layer.alpha_composite(sh.filter(ImageFilter.GaussianBlur(18)), (fx + 14 - P, fy + 18 - P))
    layer.alpha_composite(fr, (fx, fy))
    return layer, fx + hcx, fy + hcy, r


def screen_layers(R):
    D = int(2 * R + 8)
    yy, xx = np.mgrid[0:D, 0:D]; rr = np.hypot(xx - D / 2, yy - D / 2)
    g = (np.exp(-(((xx + yy) / D - 0.62) ** 2) / 0.004) * 0.10 + np.clip(1 - (xx + yy) / D, 0, 1) ** 3 * 0.10) * (rr < R + 4)
    sheen = Image.fromarray(np.dstack([np.full((D, D, 3), 255), g * 255]).astype(np.uint8), 'RGBA')
    mask = Image.new('L', (D, D), 0); ImageDraw.Draw(mask).ellipse([0, 0, D - 1, D - 1], fill=255)
    return D, sheen, mask


def watermark():
    wm = Image.new('RGBA', (W * 2, H * 2), (0, 0, 0, 0)); d = ImageDraw.Draw(wm)
    f = font(42); txt = 'PREVIEW  ·  VERTEK INNOVATIONS  ·  NOT FOR DISTRIBUTION'
    tw = d.textlength(txt, font=f)
    for i, row in enumerate(range(0, H * 2, 190)):
        xx = -(i % 2) * tw / 2
        while xx < W * 2:
            d.text((xx, row), txt, font=f, fill=(255, 255, 255, 46)); xx += tw + 80
    wm = wm.rotate(22, resample=Image.BICUBIC).crop((W // 2, H // 2, W // 2 + W, H // 2 + H))
    d = ImageDraw.Draw(wm)
    d.rounded_rectangle([30, H - 74, 620, H - 28], radius=8, fill=(0, 0, 0, 140))
    d.text((48, H - 66), 'CONCEPT PREVIEW · © Vertek Innovations', font=font(26), fill=(255, 255, 255, 220))
    return wm


def grab_frames(video, times):
    out = []
    for t in times:
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', video, '-frames:v', '1',
                              '-vf', 'scale=270:270', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                             capture_output=True, check=True).stdout
        if len(raw) == 270 * 270 * 3:
            out.append(np.frombuffer(raw, np.uint8).reshape(270, 270, 3).astype(int))
    return out


def is_static(src):
    if os.path.splitext(src)[1].lower() in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'):
        return True
    fr = grab_frames(src, [0, 1.5, 3, 4.5, 6, 7.5, 9])
    return max(np.abs(f - fr[0]).mean() for f in fr) < 1.5   # allow for encoder noise


def layers(frame):
    rng = np.random.default_rng(7)   # fixed seed: identical backdrop on every preview
    bg = bar_background(rng)
    layer, cx, cy, R = frame_layer(frame)
    D, sheen, mask = screen_layers(R)
    return bg, layer, D, sheen, mask, round(cx - D / 2), round(cy - D / 2)


def still(src, frame, out):
    bg, layer, D, sheen, mask, vx, vy = layers(frame)
    if os.path.splitext(src)[1].lower() == '.mp4':
        art = Image.fromarray(grab_frames_full(src))
    else:
        art = Image.open(src).convert('RGB')
    img = bg.convert('RGBA')
    img.paste(art.resize((D, D), Image.LANCZOS), (vx, vy), mask)
    img.alpha_composite(sheen, (vx, vy))
    img.alpha_composite(layer)
    img.alpha_composite(watermark())
    img.convert('RGB').save(out, quality=90, optimize=True)   # PIL writes no source metadata


def grab_frames_full(video):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', '1', '-i', video, '-frames:v', '1',
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    side = int(round((len(raw) / 3) ** 0.5))
    return np.frombuffer(raw, np.uint8).reshape(side, side, 3)


def video(src, frame, out):
    bg, layer, D, sheen, mask, vx, vy = layers(frame)
    tmp = tempfile.mkdtemp()
    p = lambda n: os.path.join(tmp, n)
    bg.save(p('bg.png')); layer.save(p('frame.png')); sheen.save(p('sheen.png'))
    mask.save(p('mask.png')); watermark().save(p('wm.png'))
    fc = (f"[1:v]scale={D}:{D},format=rgba[v];[2:v]format=gray[m];[v][m]alphamerge[vc];"
          f"[0:v][vc]overlay={vx}:{vy}:shortest=1[a];[a][4:v]overlay={vx}:{vy}[b];"
          f"[b][3:v]overlay=0:0[c];[c][5:v]overlay=0:0,format=yuv420p")
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-i', p('bg.png'), '-i', src,
                    '-loop', '1', '-i', p('mask.png'), '-loop', '1', '-i', p('frame.png'),
                    '-loop', '1', '-i', p('sheen.png'), '-loop', '1', '-i', p('wm.png'),
                    '-filter_complex', fc, '-t', '10', '-r', '25', '-an',
                    '-c:v', 'libx264', '-crf', '22', '-preset', 'slow', '-movflags', '+faststart',
                    '-map_metadata', '-1', '-metadata', 'title=Vertek decal concept preview',
                    '-metadata', 'copyright=Vertek Innovations - not for distribution', out], check=True)


def main(src, out_base, frame=FRAME_URL):
    frame = fetch_frame(frame, tempfile.mkdtemp())
    if is_static(src):
        out = out_base + '.jpg'; still(src, frame, out)
    else:
        out = out_base + '.mp4'; video(src, frame, out)
    print(out)


if __name__ == '__main__':
    main(*sys.argv[1:4])
