from preprocess import preprocess_image

image_path = r"C:\Ashish\FINAL YEAR PROJECT\Training\meningioma\Tr-aug-me_1.jpg"

image = preprocess_image(image_path)

print("Shape:", image.shape)
print("Data type:", image.dtype)
print("Minimum:", image.min())
print("Maximum:", image.max())