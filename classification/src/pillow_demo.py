from PIL import Image

# References:
#   Image Processing in Python with Pillow by Joyce Echessa
#     https://auth0.com/blog/image-processing-in-python-with-pillow/

if __name__ == '__main__':

    print("\nPillow")

    # Open and display an image with pillow
    # Notice that the picture opens in an external viewer
    pimage = Image.open('dept_photo.jpeg')
    print(f"Original Size:  {pimage.size}")
    pimage.show()

    # Resize to a strange size (larger or smaller is OK)
    pimage = Image.open('dept_photo.jpeg')
    pimage = pimage.resize((200, 400))
    print(f"Resized Size:   {pimage.size}")
    pimage.show()

    # Resize smaller with thumbnail function
    pimage = Image.open('dept_photo.jpeg')
    pimage.thumbnail((200, 400))
    print(f"Thumbnail Size: {pimage.size}")
    pimage.show()

    # Crop the image based on a manual bounding box
    pimage = Image.open('dept_photo.jpeg')
    pimage = pimage.crop((250, 50, 575, 400)) # top-left -> bottom-right
    print(f"Cropped Size:   {pimage.size}")
    pimage.show()

    # Convert to grayscale
    pimage = Image.open('dept_photo.jpeg')
    pimage = pimage.convert('L')
    pimage.show()
    pimage.save('gs_dept_photo.jpeg')

    pimage = Image.open('gs_dept_photo.jpeg')
    print(f"\n{pimage}\n")

    print("Done with Pillow\n")

    # -------------------------------------------------------------------------#
