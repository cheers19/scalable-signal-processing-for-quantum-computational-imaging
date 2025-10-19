# scalable-signal-processing-for-quantum-computational-imaging
Here I describe the steps for extracting **entangled photons** from a noisy, large-scale data.

## *Data Pre-processing* - Converting Pixels into Photons (see ‘process_raw.py’)

The initial stage involves converting raw pixel data from the camera into discrete, characterized photon events.

1.  We first define a time window for photon arrival and divide the whole data into **time frames**.
    We then classify each lightened batch of pixels as a "photon": the energy of the photon will be the sum of the pixels’ energies, and its position is determined by the pixel with maximal energy in the batch. Each photon is now characterized by **position, time, and energy**.
2.  We divide the camera in half and define an acceptable energy range for a photon to be considered in our count.
3.  We then **drop frames** for which:
    * There are less than 2 photons or more than 5 photons.
    * The photons’ energies are not in the acceptable energy window.

## *Nested Search Algorithm* - Extracting Entangled Photon Pairs (see ‘Camera_PDC_Scan_Area.py’ and 'Camera_PDC_Analyze.py')

4.  We define a small rectangular batch of pixels on the right half of the camera whose energy is $E_R$ and scan the left half of the camera in search for a corresponding rectangle with energy $E_L$.
   The energy condition that must be satisfied is:
   **$$|E_R + E_L - E_p| < \Delta E$$**
   where $E_p$ is the input beam energy and $\Delta E$ is the energy resolution of our system.
   We use only rectangles that contain **exactly one photon**.
5.  We **repeat** this process for several different $E_R$ energies in small increments.
6.  We then **sum** over all the counts in the rectangles that satisfied the energy condition.
    If the sum is above a certain **threshold** (dictated by background measurement) then we count the photons as ***“entangled photons”***.

# Nested search algoritm visualized

<div align="center">
  <img 
    width="500" 
    src="https://github.com/user-attachments/assets/6d48cf11-04b3-45e2-b528-bf23ade43f43" 
    alt="fig 1" 
    style="display: block; margin: 0 auto; max-width: 100%; height: auto;"
  />
</div>

# Entanglement visulized

<div align="center">
  <img 
    width="700" 
    src="https://github.com/user-attachments/assets/269317ac-5ad8-40a4-9b8f-994573770c9b" 
    alt="fig 2" 
    style="display: block; margin: 0 auto; max-width: 100%; height: auto;"
  />
</div>

<div align="center">
  <img 
    width="580" 
    src="https://github.com/user-attachments/assets/fd740aec-aae6-4f38-bb9a-b39ec345fd1f" 
    alt="fig 3" 
    style="display: block; margin: 0 auto; max-width: 100%; height: auto;"
  />
</div>

# Time analysis visulized

<div align="center">
  <img 
    width="610" 
    src="https://github.com/user-attachments/assets/c609ff3f-ab06-4461-8654-67058d5d6992" 
    alt="fig 4" 
    style="display: block; margin: 0 auto; max-width: 100%; height: auto;"
  />
</div>

