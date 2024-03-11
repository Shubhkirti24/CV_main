from PIL import Image
from scipy import fftpack
import imageio
from PIL import ImageFilter
import sys
import numpy 

if __name__ == '__main__':
     Nimage = imageio.imread(sys.argv[1], as_gray=True)
    
     fft2 = fftpack.fftshift(fftpack.fft2(Nimage))
     
     #fft2[80:87,79:92]= 0
     fft2[80:88,80:92] = 128.0
     sub_valcol = (206-78) #129 pixel
     sub_valRow = (206-90) #113 pixel
     #fft2[sub_valRow+4:sub_valcol-1,sub_valRow-2:sub_valcol-1]= 0
     fft2[120:130,115:130]=128.0


     
     
     
     ifft2 = abs(fftpack.ifft2(fftpack.ifftshift(fft2)))
     imageio.imsave('fourier_noise.png', (numpy.log(abs(fft2))* 255 /numpy.amax(numpy.log(abs(fft2)))).astype(numpy.uint8))
     imageio.imsave('Inverse.png', ifft2.astype(numpy.uint8))
