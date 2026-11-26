from pathlib import Path
import subprocess,shutil
import yaml
from PIL import Image,ImageDraw,ImageFont
out=Path("render-output");out.mkdir(exist_ok=True)
preview=Path("docs/generated");preview.mkdir(parents=True,exist_ok=True)
p=subprocess.run(["phonepilot","run","--example","examples/open_wlan.yaml","--auto-confirm-all"],capture_output=True,text=True)
(out/"execution.log").write_text(p.stdout+p.stderr,encoding="utf-8")
print(p.stdout,p.stderr);p.check_returncode()
if Path("runs").exists():shutil.copytree("runs",out/"runs",dirs_exist_ok=True)
frames=yaml.safe_load(Path("examples/open_wlan.yaml").read_text())["mock_frames"]
font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",17)
small=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",12)
im=Image.new("RGB",(900,700),"#161b22");d=ImageDraw.Draw(im)
d.text((16,12),"PhonePilot / mock WLAN example (input fixture)",font=font,fill="white")
for i,f in enumerate(frames):
    x=i*300+20;y=55
    d.rounded_rectangle((x,y,x+260,y+600),radius=18,fill="#f4f6fa")
    d.text((x+10,y+15),f"Mock frame {i+1}",font=font,fill="#152239")
    for e in f["elements"]:
        cx,cy=e["center"];xx=x+cx*260/1080;yy=y+cy*600/2400
        d.ellipse((xx-5,yy-5,xx+5,yy+5),fill="#316fea")
        d.text((xx,yy+9),e["text"],font=small,fill="#152239",anchor="mt")
im.save(out/"mock-frames.png");im.save(preview/"preview.webp",quality=90)
