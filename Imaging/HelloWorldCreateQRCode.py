# pip install qrcode
import qrcode

data = "IPCA EST MEEC\n2Ai BAIT"

img = qrcode.make(data)
img.save("qr_code.png")
