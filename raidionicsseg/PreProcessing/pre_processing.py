import logging
import os
from typing import Tuple, Any, List
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from ..Utils.volume_utilities import intensity_normalization, resize_volume, input_file_category_disambiguation
from ..Utils.io import load_nifti_volume
from .mediastinum_clipping import crop_mediastinum_volume
from .brain_clipping import crop_neuro_background
from ..Utils.configuration_parser import ConfigResources, ImagingModalityType


def prepare_pre_processing(folder: str, pre_processing_parameters: ConfigResources,
                           storage_path: str) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, np.ndarray, List[int]]:
    """
    Given the possibility to leverage multiple inputs, a sequential preprocessing is to be followed, starting with the
    first input assumed to have been prepared either manually or through the raidionics_rads_lib and to be available
    as the input0.nii.gz file.

    Parameters
    ----------
    folder: Input data folder as provided by the user
    pre_processing_parameters: Configurations parameters as provided inside the trained model folder
    storage_path: Destination folder where the results will be stored.

    Returns
    -------
    nib.Nifti1Image
        Original input volume as a nibabel object (corresponding to input0.nii.gz)
    nib.Nifti1Image
        Resampled input volume as a nibabel object, following the training configuration file, from input0.nii.gz
    np.ndarray
        Numpy array containing the complete preprocessed input, with multiple channels if multiple inputs, ready for
        inference
    List[int]
        Bounding box upper and lower limits along each axis to correspond with the background cropping strategy used
    """
    input_file = os.path.join(folder, 'input0.nii.gz')
    nib_volume, resampled_volume, data, crop_bbox, crop_mask = run_pre_processing(input_file, pre_processing_parameters,
                                                                                  storage_path)
    final_data = np.zeros((1,) + data.shape + (pre_processing_parameters.preprocessing_number_inputs,)).astype('float32')
    final_data[..., 0] = data
    for i in range(1, pre_processing_parameters.preprocessing_number_inputs):
        input_file = os.path.join(folder, 'input' + str(i) + '.nii.gz')
        _, _, data, _, _ = run_pre_processing(input_file, pre_processing_parameters, storage_path, crop_bbox)
        final_data[..., i] = data

    return nib_volume, resampled_volume, final_data, crop_bbox


def run_pre_processing(filename: str, pre_processing_parameters: ConfigResources,
                       storage_path: str, crop_bbox: None | List[int] = None,
                       crop_mask: None | np.ndarray = None) -> Tuple[nib.Nifti1Image, nib.Nifti1Image, np.ndarray, List[int], np.ndarray]:
    """
    Run the preprocessing pipeline, as specified in the pre_processing_parameters.

    Parameters
    ----------
    filename : str
        Filepath of the input volume (CT or MRI) to use.
    pre_processing_parameters : :obj:`ConfigResources`
        Loaded configuration specifying runtime parameters.
    storage_path: str
        Folder where the computed results should be stored.
    crop_bbox: None | List[int]
        In case of multiple inputs with any type of background_crop, the bounding box from input0 has to be propagated
        to the other inputs, otherwise the bounding boxes from each input would not match.
    crop_mask: None | np.ndarray
        Same reason as above, when the background cropping involves an organ mask (e.g. lungs, brain), it must be
        propagated further when multiple inputs are used.

    Returns
    -------
    nib.Nifti1Image
        Original Nifti object from loading the content of filename.
    nib.Nifti1Image
        Nifti object after conversion to a normalized space (resample_to_output).
    np.ndarray
        Fully preprocessed volume ready for inference.
    List[int]
        Indices of a bounding region within the preprocessed volume for additional cropping
         (e.g. coordinates around the brain or lungs).
         The bounding region is expressed as: [minx, miny, minz, maxx, maxy, maxz].
    None | np.ndarray
        Numpy array corresponding to the background mask when the proper background cropping strategy is selected and
        None otherwise.
    """
    logging.debug("Preprocessing - Extracting input data.")
    nib_volume = load_nifti_volume(filename)
    input_category = input_file_category_disambiguation(filename)
    processing_order = 1
    if input_category == 'Annotation':
        processing_order = 0

    logging.debug("Preprocessing - Resampling.")
    new_spacing = pre_processing_parameters.output_spacing
    if pre_processing_parameters.output_spacing == None:
        tmp = np.min(nib_volume.header.get_zooms())
        new_spacing = [tmp, tmp, tmp]

    resampled_volume = resample_to_output(nib_volume, new_spacing, order=processing_order)
    data = resampled_volume.get_fdata().astype('float32')

    logging.debug("Preprocessing - Background clipping.")
    if pre_processing_parameters.imaging_modality == ImagingModalityType.CT:
        if pre_processing_parameters.crop_background is not None and pre_processing_parameters.crop_background != 'false':
            data, crop_bbox, crop_mask = crop_mediastinum_volume(data, new_spacing, storage_path,
                                                                 pre_processing_parameters, crop_bbox, crop_mask)
    else:
        if pre_processing_parameters.crop_background is not None and \
                not pre_processing_parameters.predictions_use_preprocessed_data:
            data, crop_bbox, crop_mask = crop_neuro_background(filename, data, new_spacing, storage_path,
                                                               pre_processing_parameters, crop_bbox, crop_mask)

    if pre_processing_parameters.new_axial_size:
        logging.debug("Preprocessing - Volume resizing.")
        data = resize_volume(data, pre_processing_parameters.new_axial_size, pre_processing_parameters.slicing_plane,
                             order=processing_order)
    if input_category == 'Volume':
        # Normalize values
        logging.debug("Preprocessing - Intensity normalization.")
        data = intensity_normalization(volume=data, parameters=pre_processing_parameters)

    if pre_processing_parameters.swap_training_input:
        data = np.transpose(data, axes=(1, 0, 2))

    return nib_volume, resampled_volume, data, crop_bbox, crop_mask
