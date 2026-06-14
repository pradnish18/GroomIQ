from PIL import Image, ImageDraw
import os

os.makedirs('../hairstyles', exist_ok=True)
size = 512

# Style 1 - Short Straight Dark
img = Image.new('RGBA', (size, size), (0,0,0,0))
draw = ImageDraw.Draw(img)
draw.ellipse([100, 30, 412, 220], fill=(44,24,16,220))
draw.rectangle([80, 150, 150, 380], fill=(44,24,16,200))
draw.rectangle([362, 150, 432, 380], fill=(44,24,16,200))
img.save('../hairstyles/style1.png')
print('style1.png created')

# Style 2 - Long Wavy Brown
img = Image.new('RGBA', (size, size), (0,0,0,0))
draw = ImageDraw.Draw(img)
draw.ellipse([80, 20, 432, 230], fill=(139,69,19,220))
draw.rectangle([60, 180, 140, 420], fill=(139,69,19,200))
draw.rectangle([372, 180, 452, 420], fill=(139,69,19,200))
img.save('../hairstyles/style2.png')
print('style2.png created')

# Style 3 - Curly Afro
img = Image.new('RGBA', (size, size), (0,0,0,0))
draw = ImageDraw.Draw(img)
draw.ellipse([40, 10, 472, 320], fill=(26,10,0,210))
draw.ellipse([20, 80, 120, 200], fill=(26,10,0,190))
draw.ellipse([392, 80, 492, 200], fill=(26,10,0,190))
img.save('../hairstyles/style3.png')
print('style3.png created')

# Style 4 - Dreadlocks
img = Image.new('RGBA', (size, size), (0,0,0,0))
draw = ImageDraw.Draw(img)
draw.ellipse([100, 20, 412, 200], fill=(59,37,16,220))
for x in range(120, 400, 45):
    draw.rectangle([x, 180, x+22, 420], fill=(59,37,16,200))
img.save('../hairstyles/style4.png')
print('style4.png created')

print('All done!')
