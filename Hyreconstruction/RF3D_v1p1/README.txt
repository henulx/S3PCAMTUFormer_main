-------------------------------------------------------------------

  RF3D software for joint removal of random and fixed-pattern noise
            Public release ver. 1.1  (7 April 2020)

-------------------------------------------------------------------

Copyright (c) 2011-2020    All rights reserved.
All rights reserved.
This work should be used for nonprofit purposes only.

Authors:                     Enrique Sanchez-Monge
                             Matteo Maggioni
                             Alessandro Foi


RF3D web page:               http://www.cs.tut.fi/~foi/GCF-BM3D


-------------------------------------------------------------------
 Contents
-------------------------------------------------------------------

The package contains these files

*) DemoRF3D.m               : denoising demo script
*) LoadPSDs.m               : loads PSDs from .mat files
*) LoadRealDataFLIR.m       : reads real noisy FLIR Tau 320 data 
*) LoadRealData.m           : reads real data
*) LoadSyntheticData.m      : reads a noise-free video and adds
                              synthetic noise
*) RF3D.m                   : RF3D denoising filter [1]
*) PSDkernels.mat           : global PSDs of the random and fixed-
                              pattern noise used to generate 
                              synthetic noisy observations according to
                              FLIR Tau 320 camera noise model.
*) PSDs_bior.mat            : PSDs of the random and fixed-pattern 
                              noise components defined with respect 
                              to 2D biorthogonal wavelet transform
                              ('bior1.5' in Matlab) according to
                              FLIR Tau 320 camera noise model.
*) PSDs_dct.mat             : PSDs defined with respect to 2D dct
                              transform according to
                              FLIR Tau 320 camera noise model.
*) PSDs_haar.mat            : PSDs defined with respect to 2D haar
                              wavelet transform according to
                              FLIR Tau 320 camera noise model.

-------------------------------------------------------------------
 Usage
-------------------------------------------------------------------

Unzip RF3D.zip (contains codes) in a folder that is in the MATLAB 
path. Execute the script "DemoRF3D.m" to run the denoising demo.
You can freely change the parameters involved in the filtering by 
modifying their values in the demo script. The sequences acquired 
by a FLIR Tau 320 camera, as well as a set of noise-free test 
sequences can be downloaded from the RF3D web page. These sequences
should be used when using the provided demo.

-------------------------------------------------------------------
 Requirements
-------------------------------------------------------------------

*) Linux 64 bit, Mac OS X 64 bit, or Windows 64 bit
*) Matlab R2019b or later with installed:
   -- Image Processing Toolbox (for visualization with "implay")
   -- Wavelet Toolbox (only for non-default parameters in RF3D.m)
   [might work with earlier versions too, but not tested]

-------------------------------------------------------------------
 Change log
-------------------------------------------------------------------
v1.1   (7 April 2020)
 + improved interface of RF3D function
 + improved DemoRF3D to allow user-defined cases
 + removed video size restrictions in RF3D function
 + added 'high complexity' denoising profile for a better denoising quality
 ! fixed bug using wrong scaling factors in HT stage
 ! fixed bug in Wiener filtering that was decreasing quality of last 
   outputted frames
 ! fixed bug that limited the extent of spatio-temporal volumes to 11.

v1.0   (19 September 2014)
 + initial version
-------------------------------------------------------------------
 References
-------------------------------------------------------------------

 [1] M. Maggioni, E. Sanchez-Monge, A. Foi, "Joint removal of 
     random and fixed-pattern noise through spatiotemporal video 
     filtering", IEEE Transactions on Image Processing, vol.23, 
     no.10, pp. 4282-4296, Oct. 2014
     http://doi.org/10.1109/TIP.2014.2345261
 
 
-------------------------------------------------------------------
 Disclaimer
-------------------------------------------------------------------

Any unauthorized use of these routines for industrial or profit-
oriented activities is expressively prohibited. By downloading 
and/or using any of these files, you implicitly agree to all the 
terms of the TAU limited license, as specified in the document
LICENSE included in this package, and online at
http://www.cs.tut.fi/~foi/GCF-BM3D/legal_notice.html

-------------------------------------------------------------------
 Feedback
-------------------------------------------------------------------

If you have any comment, suggestion, or question, please do
contact
 Enrique Sanchez-Monge  < esm _at_ noiselessimaging.com >
 Alessandro Foi  < alessandro.foi _at_ tuni.fi >


