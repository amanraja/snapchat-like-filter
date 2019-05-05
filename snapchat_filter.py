import cv2
import os
import numpy as np
#path1 = os.path.join(os.path.expanduser('~'), 'Desktop','.xml')
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
face_path=r'face.xml'
mst=cv2.imread("moustache1.png")
glass=cv2.imread("glass.png")
flower=cv2.imread("flower.png")
grass=cv2.imread("green1.png")
face_cascade = cv2.CascadeClassifier(face_path)
cap = cv2.VideoCapture(0)
rmst=np.array(mst)
def cv_min_put(img,msk):
    ## img and msk should of same size
    pass
def put_moustache(mst,fc,w,h):
    x=int(0.3*w)
    y=int(0.65*h)
    mw = int(w*0.4)+1
    mh = int(h*0.15)+1
    mst = cv2.resize(mst,(mw,mh))
    part=fc[y:y+mh,x:x+mw,:].copy()
    #part[:,:,:]=255
    for i in range(mh):
        for j in range(mw):
            if mst[i,j,1]!=255:
                part[i,j,:]=mst[i,j,:]
    fc[y:y+mh,x:x+mw,:]=part
    return fc
def put_glass(glass,fc,w,h):
    x=int(0.15*w)
    y=int(0.3*h)
    mw = int(w*0.7)+1
    mh = int(h*0.2)+1
    glass = cv2.resize(glass,(mw,mh))
    part=fc[y:y+mh,x:x+mw,:].copy()
    #part[:,:,:]=255
    for i in range(mh):
        for j in range(mw):
            if glass[i,j,1]!=255:
                part[i,j,:]=glass[i,j,:]
    fc[y:y+mh,x:x+mw,:]=part
    return fc
'''def put_flower(flower,img,x,y,w,h):
    if x-100>0 and y-100>0 and (x+int(w*1.5)+1)<640 and (y+int(h*1.2)+1)<480:
        mw = int(w*1.5)+1
        mh = int(h*1.2)+1
        part=img[y-100:y+mh,x-100:x+mw,:]
        a,b,c=part.shape
        flower = cv2.resize(flower,(b,a))
        #part[:,:,:]=flower
        for i in range(a):
            for j in range(b):
                if flower[i,j,1]<=230:
                    part[i,j,:]=flower[i,j,:]
    #img[y-40:y+mh,x-100:x+mw,:]=part
    return img'''
def put_flower(flower,img,x,y,w,h):
    flower = cv2.resize(flower,(int(w*1.5),int(h*1.5)))
    for i in range(int(h*1.5)):
            for j in range(int(w*1.5)):
                if flower[i,j,1]<=230:
                    img[y+i-int(h*.6),x+j-int(w*.2),:]=flower[i,j,:]
    return img
def put_grass(grass,img,x,y,w,h):
    grass = cv2.resize(grass,(int(w*1.5),int(h*1.5)))
    for i in range(int(h*1.5)):
            for j in range(int(w*1.5)):
                if grass[i,j,1]<=230:
                    img[y+i-int(h*.7),x+j-int(w*.3),:]=grass[i,j,:]
    return img
value=0
count=0
while 1:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    x=0
    y=0
    w=0
    h=0
    for (xf,yf,wf,hf) in faces[:1]:
        roi_face=img[yf:yf+hf,xf:xf+wf]
        x,y,w,h=xf,yf,wf,hf
        #roi_face= put_moustache(mst,roi_face,wf,hf)
        #roi_face= put_glass(glass,roi_face,wf,hf)
        #roi_face= put_flower(flower,img,xf,yf,wf,hf)
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1
    #print(value)    
    if w>0 and h>0:
        if value==113:
            img=put_flower(flower,img,x,y,w,h)
        elif value==119:
            img=put_grass(grass,img,x,y,w,h)
        elif value==101:
            roi_face= put_moustache(mst,roi_face,w,h)
        elif value==114:
            roi_face= put_glass(glass,roi_face,w,h)
        else:
            pass
    cv2.imshow('window',img)
    k = cv2.waitKey(30) & 0xff
    if k == ord('a'):
        break
    if k==ord('s'):
        count=count+1
        name="frame"+str(count)+".jpg"
        cv2.imwrite(name,img)
    if k==ord('q') or k==ord('w') or k==ord('e') or k==ord('r'):
        value=k
cap.release()
cv2.destroyAllWindows()
