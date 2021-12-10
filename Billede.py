# Der importeres nødvendige biblioteker
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
from imageio import imread
from numpy.fft import fft,fft2, ifft2, fftshift, ifftshift

I = imread("image.jpg", as_gray=True)  # Billede importeres i sort-hvid

# Billede vises
plt.figure(figsize=(8,8))                   
plt.imshow(I, cmap="gray")  
plt.savefig('FFT1.png', dpi=200) 
plt.show()

fI = fft2(I) # Den 2-dimensionelle FFT beregnes 

# Plot af billedet samt logaritmen af dets korresponderende frekvensdomæne 
plt.figure(figsize=(12,6))                  
plt.subplot(1,2,1) ; plt.title("Billede")     
plt.imshow(I, cmap="gray")               
plt.subplot(1,2,2) ; plt.title("Frekvensdomæne")     
# Af visuelle grunde tages logaritmen af FFT-værdierne. 
# Herudover centreres FFT-værdierne.
plt.imshow(fftshift(np.log(1e-5 + abs(fI))), cmap="gray")
plt.savefig('FFT2.png', dpi=200) 
plt.show()

fI_old = fI.copy() # Kopi af 2FFT-værdierne før filtrering
fI[np.abs(fI) < 200000] = 0 # Filtrerer alle værdier under 200000

# Plot af fouriertransformen frekvensdomæne før og efter filtrering
plt.figure(figsize=(12,6))                  
plt.subplot(1,2,1) ; plt.title("Før filtrering")      
plt.imshow(fftshift(np.log(1e-5 + abs(fI_old))), cmap="gray")              
plt.subplot(1,2,2) ; plt.title("Efter filtrering")   
plt.imshow(fftshift(np.log(1e-5 + abs(fI))), cmap="gray")          
plt.savefig('FFT3.png', dpi=200) 
plt.show()

fI_compressed = ifft2(fI) # Den inverse 2-dimensionelle FFT beregnes 

# Plot af billede før og efter filtrering
plt.figure(figsize=(12,6))                
plt.subplot(1,2,1) ; plt.title("Før filtrering")    
plt.imshow(I, cmap="gray")                
plt.subplot(1,2,2) ; plt.title("Efter filtrering")       
plt.imshow(abs(fI_compressed), cmap="gray")       
plt.savefig('FFT4.png', dpi=200) 
plt.show()

# Sammenligning af antallet af non-zero komponenter i FFT arrayet før og efter filtrering
print("Før: ", np.count_nonzero(fI_old)) # Produktet af dimensionerne på billedet 
print("Efter: ", np.count_nonzero(fI)) # Efter filtrering
