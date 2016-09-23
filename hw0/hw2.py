from PIL import Image
import sys


sourceim = Image.open(sys.argv[1])
destim = sourceim.rotate(180)
destim.save("ans2.png")