from setuptools import setup


setup(name='driving_uncertainty',
      version="0.0",
      packages=['driving_uncertainty', 
                'driving_uncertainty.data_preparation', 
                'driving_uncertainty.image_dissimilarity',
                'driving_uncertainty.image_segmentation',
                'driving_uncertainty.image_synthesis',
                'driving_uncertainty.options',
                'driving_uncertainty.image_synthesis_spade',
                ])
