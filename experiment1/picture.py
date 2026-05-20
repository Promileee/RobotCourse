import cv2
import matplotlib.pyplot as plt

img_path = r'C:\Users\20230\Desktop\RobotCourse\experiment1\photo1.png'
img = cv2.imread(img_path)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

r,g,b = cv2.split(img_rgb)

plt.figure(figsize=(10,10))
plt.subplot(2,3,1)
plt.imshow(img_rgb)

plt.subplot(2,3,2)
plt.imshow(img_gray, cmap='gray')

plt.subplot(2,3,4)
plt.imshow(r, cmap='Reds')

plt.subplot(2,3,5)
plt.imshow(g, cmap='Greens')

plt.subplot(2,3,6)
plt.imshow(b, cmap='Blues')

plt.show()