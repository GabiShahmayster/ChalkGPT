import dataclasses
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike
from typing import List, Optional, Tuple, Dict

import cv2
from cv2 import DMatch, KeyPoint
import numpy as np
import torch

@dataclass
class KeypointMatchingResults:
    """
    A data class to hold the results of keypoint matching between two images.

    Attributes
    ----------
    matches : List[DMatch]
        A list of matched keypoint pairs between the train and query images.
    train_keypoints : List[KeyPoint]
        Keypoints detected in the train (or reference) image.
    query_keypoints : List[KeyPoint]
        Keypoints detected in the query image.
    train_image : np.array
        The train (or reference) image in which the train keypoints were detected.
    query_image : np.array
        The query image in which the query keypoints were detected.
    """
    matches: List[DMatch]
    train_keypoints: List[KeyPoint]
    query_keypoints: List[KeyPoint]
    train_image: np.array
    query_image: np.array

    @classmethod
    def load_from_file(cls, filename: PathLike, device: torch.device):
        load_dict = torch.load(filename, map_location=device)
        data_dict = {}
        for k, v in load_dict.items():
            if k == "train_image" or k == "query_image":
                data_dict[k] = v
            elif k == "matches":
                data_dict[k] = get_tensor_to_dmatches(v)
            elif k == "train_keypoints" or k == "query_keypoints":
                data_dict[k] = get_opencv_keypoint_from_tensor(v)
        return KeypointMatchingResults(**data_dict)

    def save(self, filename: os.PathLike):
        save_dict: Dict = {"train_image": self.train_image,
                           "query_image": self.query_image,
                           "train_keypoints": get_tensor_from_opencv_keypoint(self.train_keypoints),
                           "query_keypoints": get_tensor_from_opencv_keypoint(self.query_keypoints),
                           "matches":get_tensor_from_opencv_dmatches(self.matches),
                           }
        torch.save(save_dict, filename)

    def __eq__(self, other):
        if not isinstance(other, KeypointMatchingResults):
            raise TypeError("Can only compare to objects of type KeypointMatchingResults")

        if len(self.matches) != len(other.matches):
            return False
        if len(self.train_keypoints) != len(other.train_keypoints):
            return False
        if len(self.query_keypoints) != len(other.query_keypoints):
            return False
        if self.train_image.shape != other.train_image.shape:
            return False
        if self.query_image.shape != other.query_image.shape:
            return False
        if np.any(self.train_image != other.train_image):
            return False
        if np.any(self.query_image != other.query_image):
            return False

        self_matches_sorted = sorted(self.matches, key=lambda x: (x.queryIdx, x.trainIdx))
        other_matches_sorted = sorted(other.matches, key=lambda x: (x.queryIdx, x.trainIdx))

        dmatch_attributes = ['distance', 'imgIdx', 'queryIdx', 'trainIdx']
        for dmacth_self, dmatch_other in zip(self_matches_sorted, other_matches_sorted):
            for attribute in dmatch_attributes:
                if getattr(dmacth_self, attribute) != getattr(dmatch_other, attribute):
                    return False

        keypoint_attributes = ['angle', 'class_id', 'octave', 'pt', 'response', 'size']

        self_query_keypoints_sorted = sorted(self.query_keypoints, key=lambda x: (x.pt[0], x.pt[1]))
        other_query_keypoints_sorted = sorted(other.query_keypoints, key=lambda x: (x.pt[0], x.pt[1]))

        for keypoint_self, keypoint_other in zip(self_query_keypoints_sorted, other_query_keypoints_sorted):
            for attribute in keypoint_attributes:
                if getattr(keypoint_self, attribute) != getattr(keypoint_other, attribute):
                    return False

        self_train_keypoints_sorted = sorted(self.train_keypoints, key=lambda x: (x.pt[0], x.pt[1]))
        other_train_keypoints_sorted = sorted(other.train_keypoints, key=lambda x: (x.pt[0], x.pt[1]))

        for keypoint_self, keypoint_other in zip(self_train_keypoints_sorted, other_train_keypoints_sorted):
            for attribute in keypoint_attributes:
                if getattr(keypoint_self, attribute) != getattr(keypoint_other, attribute):
                    return False

        return True

    def __getstate__(self):
        matches = self.matches
        query_keypoints = self.query_keypoints
        train_keypoints = self.train_keypoints

        matches_unpacked = [
            {'_distance': dmatch.distance,
             '_imgIdx': dmatch.imgIdx,
             '_queryIdx': dmatch.queryIdx,
             '_trainIdx': dmatch.trainIdx}
            for dmatch in matches]

        query_keypoints_unpacked = [
            {'angle': keypoint.angle,
             'class_id': keypoint.class_id,
             'octave': keypoint.octave,
             'x': keypoint.pt[0],
             'y': keypoint.pt[1],
             'response': keypoint.response,
             'size': keypoint.size}
            for keypoint in query_keypoints]

        train_keypoints_unpacked = [
            {'angle': keypoint.angle,
             'class_id': keypoint.class_id,
             'octave': keypoint.octave,
             'x': keypoint.pt[0],
             'y': keypoint.pt[1],
             'response': keypoint.response,
             'size': keypoint.size}
            for keypoint in train_keypoints]

        return (matches_unpacked,
                self.query_image,
                query_keypoints_unpacked,
                self.train_image,
                train_keypoints_unpacked)

    def __setstate__(self, state):
        matches_unpacked, query_image, query_keypoints_unpacked, train_image, train_keypoints_unpacked = state

        self.matches = [cv2.DMatch(**match) for match in matches_unpacked]
        self.train_keypoints = [cv2.KeyPoint(**keypoint) for keypoint in train_keypoints_unpacked]
        self.train_image = train_image
        self.query_keypoints = [cv2.KeyPoint(**keypoint) for keypoint in query_keypoints_unpacked]
        self.query_image = query_image



@dataclass
class KeypointData:
    """
    A data class to hold the images, keypoints, and descriptors for an image, both in OpenCV format and as PyTorch tensors.
    """
    image: np.array

    _keypoints: Optional[List[KeyPoint]] = None
    _descriptors: Optional[np.array] = None
    _scores: Optional[np.array] = None
    _scales: Optional[np.array] = None
    _angles: Optional[np.array] = None

    _keypoints_tensor: Optional[torch.Tensor] = None
    _descriptors_tensor: Optional[torch.Tensor] = None
    _scores_tensor: Optional[torch.Tensor] = None
    _scales_tensor: Optional[torch.Tensor] = None
    _angles_tensor: Optional[torch.Tensor] = None

    def __init__(self, image: np.ndarray, keypoints: Optional[List[KeyPoint]] = None,
                 descriptors: Optional[np.ndarray] = None, score: Optional[np.ndarray] = None,
                 keypoints_tensor: Optional[torch.Tensor] = None, descriptors_tensor: Optional[torch.Tensor] = None,
                 scores_tensor: Optional[torch.Tensor] = None):
        self.image = image

        if isinstance(keypoints, list) or keypoints is None:
            self._keypoints = keypoints
        else:
            raise ValueError(f'Expected list of KeyPoints or None, got {type(keypoints)}.')

        if isinstance(descriptors, np.ndarray) or descriptors is None:
            self._descriptors = descriptors
        else:
            raise ValueError(f'Expected np.ndarray or None, got {type(descriptors)}.')

        if isinstance(score, np.ndarray) or score is None:
            self._scores = score
        else:
            raise ValueError(f'Expected np.ndarray or None, got {type(score)}.')

        if isinstance(keypoints_tensor, torch.Tensor) or keypoints_tensor is None:
            self._keypoints_tensor = keypoints_tensor
        else:
            raise ValueError(f'Expected torch.Tensor or None, got {type(keypoints_tensor)}.')

        if isinstance(descriptors_tensor, torch.Tensor) or descriptors_tensor is None:
            self._descriptors_tensor = descriptors_tensor
        else:
            raise ValueError(f'Expected torch.Tensor or None, got {type(descriptors_tensor)}.')

        if isinstance(scores_tensor, torch.Tensor) or scores_tensor is None:
            self._scores_tensor = scores_tensor
        else:
            raise ValueError(f'Expected torch.Tensor or None, got {type(scores_tensor)}.')

    def remove_lower_median_of_scores(self):
        median_tensor = torch.median(self.score_tensor) if self.score_tensor is not None else None
        median = None
        if median_tensor is not None:
            indices_to_keep_tensor = torch.where(self.score_tensor > median_tensor)
        if median is not None:
            indices_to_keep = torch.where(self.score > median)
        # update non-tensor arrays
        if median_tensor is not None:
            self.score_tensor = self.score_tensor[indices_to_keep_tensor]
            self.keypoints_tensor = self.keypoints_tensor[indices_to_keep_tensor]
            self.descriptors_tensor = self.descriptors_tensor[indices_to_keep_tensor]

    def save_to_file(self, filename: str):
        save_dict: Dict = {"image": self.image,
                           "keypoints_tensor": self.keypoints_tensor,
                           "descriptors_tensor": self.descriptors_tensor,
                           "scores_tensor": self.score_tensor}
        torch.save(save_dict, filename)

    @classmethod
    def load_from_file(cls, filename: PathLike, device: torch.device):
        load_dict = torch.load(filename, map_location=device)
        data_device = {}
        for k, v in load_dict.items():
            if k == "image":
                data_device[k] = v
            else:
                data_device[k] = v.to(device)
        return KeypointData(**data_device)

    @property
    def keypoints(self):
        if self._keypoints is None and self._keypoints_tensor is not None:
            self._keypoints = get_opencv_keypoint_from_tensor(self._keypoints_tensor)
        return self._keypoints

    @keypoints.setter
    def keypoints(self, value):
        self._keypoints = value

    @property
    def keypoints_tensor(self):
        if self._keypoints_tensor is None and self._keypoints is not None:
            self._keypoints_tensor = get_tensor_from_opencv_keypoint(self._keypoints)
        return self._keypoints_tensor

    @keypoints_tensor.setter
    def keypoints_tensor(self, value):
        self._keypoints_tensor = value

    @property
    def descriptors(self):
        if self._descriptors is None and self._descriptors_tensor is not None:
            self._descriptors = self._descriptors_tensor.detach().cpu().numpy()
        return self._descriptors

    @descriptors.setter
    def descriptors(self, value):
        self._descriptors = value

    @property
    def descriptors_tensor(self):
        if self._descriptors_tensor is None and self._descriptors is not None:
            self._descriptors_tensor = torch.tensor(self._descriptors)
        return self._descriptors_tensor

    @descriptors_tensor.setter
    def descriptors_tensor(self, value):
        self._descriptors_tensor = value

    @property
    def score(self):
        if self._scores is None:
            if self._scores_tensor is not None:
                self._scores = self._scores_tensor.numpy()
            elif self._keypoints is not None:
                self._scores = get_score_array_from_opencv_keypoint(self._keypoints)
        return self._scores

    @score.setter
    def score(self, value):
        self._scores = value

    @property
    def score_tensor(self):
        if self._scores_tensor is None:
            if self._scores is not None:
                self._scores_tensor = torch.tensor(self._scores)
            elif self._keypoints is not None:
                self._scores = get_score_array_from_opencv_keypoint(self._keypoints)
                self._scores_tensor = torch.tensor(self._scores)
                
        return self._scores_tensor

    @property
    def scales_tensor(self):
        if self._scales_tensor is None:
            if self._scales is not None:
                self._scales_tensor = torch.tensor(self._scales)
            elif self._keypoints is not None:
                self._scales = get_scale_array_from_opencv_keypoint(self._keypoints)
                self._scales_tensor = torch.tensor(self._scales)
        return self._scales_tensor
    
    @property
    def angles_tensor(self):
        if self._angles_tensor is None:
            if self._angles is not None:
                self._angles_tensor = torch.tensor(self._angles)
            elif self._keypoints is not None:
                self._angles = get_angle_array_from_opencv_keypoint(self._keypoints)
                self._angles_tensor = torch.tensor(self._angles)
                
        return self._angles_tensor

    @score_tensor.setter
    def score_tensor(self, value):
        self._scores_tensor = value

class MatcherBaseClass(ABC):
    """
    Base class for keypoint matchers.

    This class serves as a blueprint for keypoint matcher implementations. It defines the
    interface for matching keypoints between two images and visualizing the matches.

    Methods
    -------
    match(kp1_data: KeypointData, kp2_data: KeypointData) -> KeypointMatchingResults:
        Abstract method for matching keypoints between two images based on the given input.

    draw_matches(matching_results: KeypointMatchingResults, n_skip: int = 0) -> np.array:
        Visualizes the matched keypoints between two images.

    Notes
    -----
    Subclasses should provide concrete implementations of the abstract methods defined in this v0 class.
    """

    @abstractmethod
    def match(self, query_kp_data: KeypointData, train_kp_data: KeypointData) -> KeypointMatchingResults:
        """
        Match keypoints and descriptors from two sets of image data.

        Parameters
        ----------
        query_kp_data : KeypointData
            An instance of the KeypointData class containing the image, keypoints, descriptors,
            and their tensor representations for the query image.

        train_kp_data : KeypointData
            An instance of the KeypointData class containing the image, keypoints, descriptors,
            and their tensor representations for the train image.

        Returns
        -------
        KeypointMatchingResults
            A data class containing the results of the keypoint matching between the two images.

        Notes
        -----
        This is an abstract method and should be implemented in subclasses.
        """
        pass

    def draw_matches(self, matching_results: KeypointMatchingResults, train_image: np.ndarray = None, query_image: np.ndarray = None) -> np.array:
        """
        Draw matches between keypoints of two images using the provided matching results.

        Parameters
        ----------
        matching_results : KeypointMatchingResults
            The results of keypoint matching which includes matched pairs, keypoints, and images.
        n_skip : int, optional
            Number of matches to skip for visualization (default is 0, meaning all matches are drawn).
        train_image: np.ndarray, optional
            Will be used instead of grayscale image from matching_results
        query_image: np.ndarray, optional
            Will be used instead of grayscale image from matching_results
        Returns
        -------
        np.array
            An image with the matched keypoints connected by lines between the train and query images.

        Notes
        -----
        The function converts grayscale images to BGR before drawing matches for visualization purposes.
        """
        list_of_matches: List[DMatch] = matching_results.matches
        query_keypoints: List[KeyPoint] = matching_results.query_keypoints
        train_keypoints: List[KeyPoint] = matching_results.train_keypoints

        if train_image is None:
            train_image: np.ndarray = cv2.cvtColor(matching_results.train_image, cv2.COLOR_GRAY2BGR)
        if query_image is None:
            query_image: np.ndarray = cv2.cvtColor(matching_results.query_image, cv2.COLOR_GRAY2BGR)
        matches_img: np.ndarray = cv2.drawMatches(query_image, query_keypoints,train_image, train_keypoints,
                                                  list_of_matches, None)
        return matches_img


class DetectorAndExtractor(ABC):
    """
    Abstract v0 class for keypoint detectors and descriptor extractors.

    This class serves as a blueprint for implementations of keypoint detection and descriptor
    extraction algorithms. It defines the interface for detecting keypoints in an image and computing
    their corresponding descriptors.

    Methods
    -------
    detectAndCompute(image: np.array, mask: np.ndarray, **kwargs) -> Tuple[List[KeyPoint], np.ndarray]:
        Abstract method to detect keypoints in the provided image and compute their descriptors.

    Notes
    -----
    Subclasses should provide concrete implementations of the abstract methods defined in this v0 class.
    """

    @abstractmethod
    def detectAndCompute(self, image: np.array, mask: np.ndarray = None, **kwargs) -> KeypointData:
        """
        Detect keypoints in the given image and compute their descriptors.

        Parameters
        ----------
        image : np.array
            The input image for keypoint detection and descriptor extraction.
        mask : np.ndarray, optional
            An optional mask to specify regions of the image where keypoints should be detected.
        **kwargs
            Additional keyword arguments for specific implementations of detectors and extractors.

        Returns
        -------
        Tuple[List[KeyPoint], np.ndarray]
            A tuple containing a list of detected keypoints and a NumPy array of computed descriptors.

        Notes
        -----
        This is an abstract method and should be implemented in subclasses.
        """
        pass


def get_opencv_keypoint_from_tensor(tensor: torch.tensor) -> List[cv2.KeyPoint]:
    """
    Convert a PyTorch tensor of keypoints to a list of OpenCV KeyPoint objects.

    Parameters
    ----------
    tensor : torch.tensor
        A tensor containing keypoints. The tensor is expected to have keypoints in its rows with
        x and y coordinates in the columns.

    Returns
    -------
    List[cv2.KeyPoint]
        A list of OpenCV KeyPoint objects corresponding to the input keypoints.

    Notes
    -----
    """
    # TODO: handle different input types?
    kpts = tensor.detach().cpu().numpy()
    out = get_opencv_keypoint_from_np_array(kpts)
    return out


def get_opencv_keypoint_from_np_array(kpts: np.ndarray) -> List[cv2.KeyPoint]:
    """
    Convert a NumPy array of keypoints to a list of OpenCV KeyPoint objects.

    Parameters
    ----------
    kpts : np.ndarray
        A NumPy array containing keypoints. The array is expected to have keypoints in its rows with
        x and y coordinates in the columns.

    Returns
    -------
    List[cv2.KeyPoint]
        A list of OpenCV KeyPoint objects corresponding to the input keypoints.
    """
    out: List[cv2.KeyPoint] = []
    for kpt in kpts:
        kpt_opencv: cv2.KeyPoint = cv2.KeyPoint()
        kpt_opencv.pt = (kpt[0], kpt[1])
        out.append(kpt_opencv)
    return out


def get_np_array_from_opencv_keypoint(keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    """
    Convert a list of OpenCV KeyPoint objects to a NumPy array.

    Parameters
    ----------
    keypoints : List[cv2.KeyPoint]
        A list of OpenCV KeyPoint objects.

    Returns
    -------
    np.ndarray
        A NumPy array with keypoints in its rows and x and y coordinates in the columns.
    """
    kpts = np.zeros((len(keypoints), 2), dtype=np.float32)
    for i, kpt in enumerate(keypoints):
        kpts[i, :] = [kpt.pt[0], kpt.pt[1]]
    return kpts

def get_score_array_from_opencv_keypoint(keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    return np.array([k.response for k in keypoints], dtype=np.float32)

def get_scale_array_from_opencv_keypoint(keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    return np.array([k.size for k in keypoints], dtype=np.float32)

def get_angle_array_from_opencv_keypoint(keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    return np.deg2rad(np.array([k.angle for k in keypoints], dtype=np.float32))

def get_tensor_from_opencv_keypoint(keypoints: List[cv2.KeyPoint]) -> torch.tensor:
    """
    Convert a list of OpenCV KeyPoint objects to a PyTorch tensor.

    Parameters
    ----------
    keypoints : List[cv2.KeyPoint]
        A list of OpenCV KeyPoint objects.

    Returns
    -------
    torch.tensor
        A tensor containing keypoints with keypoints in its rows and x and y coordinates in the columns.
    """
    kpts = get_np_array_from_opencv_keypoint(keypoints)
    return torch.from_numpy(kpts)


def get_tensor_from_opencv_dmatches(dmatches: List[cv2.DMatch]) -> torch.Tensor:
    """
    Convert a list of OpenCV DMatch objects to a PyTorch tensor.

    Parameters
    ----------
    dmatches : List[cv2.DMatch]
        A list of OpenCV DMatch objects.

    Returns
    -------
    torch.Tensor
        A tensor of shape (N, 4) where N is the number of DMatch objects,
        and each row contains [queryIdx, trainIdx, imgIdx, distance].
    """
    # Create a list of tuples from DMatch attributes
    match_data = [(dm.queryIdx, dm.trainIdx, dm.imgIdx, dm.distance) for dm in dmatches]

    # Convert the list of tuples to a NumPy array
    match_array = np.array(match_data, dtype=np.float32)

    # Convert the NumPy array to a PyTorch tensor
    match_tensor = torch.from_numpy(match_array)

    return match_tensor

def get_tensor_to_dmatches(match_tensor: torch.Tensor) -> List[cv2.DMatch]:
    """
    Convert a PyTorch tensor to a list of OpenCV DMatch objects.

    Parameters
    ----------
    match_tensor : torch.Tensor
        A tensor of shape (N, 4) where N is the number of matches,
        and each row contains [queryIdx, trainIdx, imgIdx, distance].

    Returns
    -------
    List[cv2.DMatch]
        A list of OpenCV DMatch objects corresponding to the input tensor.
    """
    # Ensure the tensor is on the CPU and convert it to a NumPy array
    match_array = match_tensor.cpu().detach().numpy()

    # Create a list of DMatch objects from the NumPy array
    dmatches = [cv2.DMatch(_queryIdx=int(row[0]), _trainIdx=int(row[1]), _imgIdx=int(row[2]), _distance=row[3]) for row in match_array]

    return dmatches