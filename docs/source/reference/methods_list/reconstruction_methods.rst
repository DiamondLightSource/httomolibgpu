.. _image_reconstruction_module:

Image reconstruction methods
****************************

Methods from :mod:`httomolibgpu.recon.algorithm` module are required to reconstruct the projection data, i.e., converting the set of sinograms into a reconstructed volume.
The reconstruction methods can be divided into two groups: Direct analytical methods (:ref:`method_LPRec3d_tomobar`, :ref:`method_FBP3d_tomobar`, :ref:`method_FBP2d_astra`) and Iterative methods. 
The former are faster and suitable for the majority of well-sampled and well-exposed data. 
The latter are more complex and slower methods suitable for erroneous or/and undersampled data. 

.. toctree::
   :maxdepth: 2

   reconstruction/LPRec3d_tomobar
   reconstruction/FBP3d_tomobar
   reconstruction/FBP2d_astra
   reconstruction/SIRT3d_tomobar
   reconstruction/CGLS3d_tomobar


