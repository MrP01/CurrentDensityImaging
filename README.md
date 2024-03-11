# Magnetic Resonance Current Density Imaging (MRCDI)
This lab rotation project was concerned with the development, exploration and implementation of a new optimisation approach for the reconstruction of current density / conductivity of tissue within a Magnetic Resonance Imaging (MRI) setup.

The project was split into two steps: First, simulating the magnetic field modulation given a current density field inside a *phantom* (the forward procedure).
And second, reconstructing current density (including conductivity) using a novel optimisation model, based on measurements of the z-component of the magnetic field, with insight gained from the forward procedure (which, in turn is referred to as the backward procedure).

The implementation was built on top of KomaMRI and is available in this repository.
An exemplary reconstruction is documented in the results section of [the report](https://raw.githubusercontent.com/MrP01/CurrentDensityImaging/master/report/report.pdf).
