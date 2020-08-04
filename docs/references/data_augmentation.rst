Data Augmentation
=================

.. module:: d3rlpy.augmentation

d3rlpy provides data augmentation techniques tightly integrated with
reinforcement learning algorithms.

`Kostrikov et al., Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels. <https://arxiv.org/abs/2004.13649>`_

 Efficient data augmentation can boost algorithm performance significantly. 

.. code-block:: python

    from d3rlpy.algos import DiscreteCQL

    # choose data augmentation types
    cql = CQL(augmentation=['random_shift', 'intensity'],
              n_augmentations=2)

You can also tune data augmentation parameters by yourself.

.. code-block:: python

    from d3rlpy.augmentation.image import RandomShift

    random_shift = RandomShift(shift_size=10)

    cql = CQL(augmentation=[random_shift, 'intensity'],
              n_augmentations=2)

Image Observation
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.augmentation.image.RandomShift
   d3rlpy.augmentation.image.Cutout
   d3rlpy.augmentation.image.HorizontalFlip
   d3rlpy.augmentation.image.VerticalFlip
   d3rlpy.augmentation.image.RandomRotation
   d3rlpy.augmentation.image.Intensity
