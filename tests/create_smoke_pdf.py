from pathlib import Path

import fitz


output = Path("data/input/smoke_test.pdf")
output.parent.mkdir(parents=True, exist_ok=True)
document = fitz.open()
page = document.new_page()
page.insert_text((72, 72), "WARNING")
page.insert_textbox((72, 100, 500, 140), "Setting data modification for backup of dialogs originating from MMS entails restarting the\nserver application.")
page.insert_text((72, 170), "Value: 12345")
page.insert_text((72, 190), "A2-A-RF0181")
page.insert_text((72, 220), "Server Configuration    MMS Interface")
page.insert_text((72, 250), "Item        Status")
page.insert_text((72, 265), "Database   OK")
pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(400, 300, 440, 340), 0)
pixmap.clear_with(0xB8D8E8)
page.insert_image(fitz.Rect(400, 300, 440, 340), pixmap=pixmap)
document.new_page()
document.save(output)
document.close()
print(output)
