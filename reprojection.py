import numpy as np
import cv2

img = cv2.imread('360.jpg')

height, width = img.shape[:2]

print ('width = ', width , ', height = ', height)

#Two variables
yaw = np.pi/4 
pitch = np.pi/4;
roll = 0
#target image coordinates in the dst image
# u = np.linspace(np.pi/2 , np.pi * 3/ 2, width/2)
# v = np.linspace(np.pi/4, np.pi*3/4, height/2)
# u = np.linspace(width/4 + 0.5 , width * 3/4 - 0.5, width/2) *np.pi * 2 / width 
# v = np.linspace(height/4 + 0.5, height * 3/4 - 0.5, height/2) * np.pi / height
u = np.linspace(0.5 , width  - 0.5, width) *np.pi * 2 / width 
v = np.linspace( 0.5, height - 0.5, height) * np.pi / height


# x = np.outer(np.cos(u), np.sin(v))
# y = np.outer(np.sin(u), np.sin(v))
# z = np.outer(np.ones(np.size(u)), np.cos(v))

#Convert the image coordinates to coordinates in 3D
x = np.outer(np.sin(v), np.cos(u))
y = np.outer(np.sin(v), np.sin(u))
z = np.outer(np.cos(v), np.ones(np.size(u)))
 
 
#rotation angles
a = yaw        # equal to yaw or phi
b = pitch    # equal to pitch or theta 
r = roll        # equal to roll or psi

#rotation matrix
rot_a = np.array([np.cos(a)* np.cos(b), np.cos(a)*np.sin(b)*np.sin(r) - np.sin(a)*np.cos(r), np.cos(a)*np.sin(b)*np.cos(r) + np.sin(a)*np.sin(r) ])
rot_b = np.array([np.sin(a)* np.cos(b), np.sin(a)*np.sin(b)*np.sin(r) + np.cos(a)*np.cos(r), np.sin(a)*np.sin(b)*np.cos(r) - np.cos(a)*np.sin(r) ])
rot_c = np.array([-np.sin(b), np.cos(b)*np.sin(r) , np.cos(b)*np.cos(r) ])

#rotate the image to the correct place
xx = rot_a[0] * x + rot_a[1] * y + rot_a[2] * z
yy = rot_b[0] * x + rot_b[1] * y + rot_b[2] * z
zz = rot_c[0] * x + rot_c[1] * y + rot_c[2] * z
zz = np.clip(zz, -1, 1)

#calculate the (u, v) in the original equirectangular map
map_u = (((np.arctan2(yy, xx) + 2* np.pi)) % (2*np.pi)) * width / (2* np.pi) - 0.5
map_v = np.arccos(zz) * height/ np.pi - 0.5

dstMap_u, dstMap_v = cv2.convertMaps(map_u.astype(np.float32), map_v.astype(np.float32), cv2.CV_16SC2)
# remap 
#dst_img = cv2.remap(img, map_u.astype(np.float32), map_v.astype(np.float32), cv2.INTER_LINEAR )
dst_img = cv2.remap(img, dstMap_u, dstMap_v, cv2.INTER_LINEAR )

#cv2.imshow('image_orig', img)
cv2.imshow('image',dst_img)
# cv2.imshow('image croped', cmp_img)
# cv2.imshow('diff', cv2.absdiff(dst_img,cmp_img))
# print (np.sum(cv2.absdiff(dst_img,cmp_img)))

cv2.waitKey(0)
cv2.destroyAllWindows()
