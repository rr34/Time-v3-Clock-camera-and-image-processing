from PIL import Image, PngImagePlugin

png_file_1 = Image.open('slayer_cal_image_with_bigtxtdictionary2.png')


"""
camera_data_retiever = PngImagePlugin.PngInfo()

camera_data_from_png = camera_data_retiever.read(png_file_1)
"""

png_text_dictionary = png_file_1.text

print(png_text_dictionary['Az / Alt Model'])