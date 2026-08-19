# VoronoiCity3

This repository is the third installment in the “Automatic City Generation” series.
In [VoronoiCity2](https://github.com/taKana671/VoronoiCity2.git), I created buildings by generating convex polygonal prisms from Voronoi-divided cells. 
For this third installment, I created buildings using the vertex coordinates of rounded Voronoi cells, resulting in a city that coexists with nature.
As a first attempt, I used shaders to generate a large amount of vegetation, including grass and flowers, and also used shaders to animate the water and flowers. 
The distribution of the vegetation is calculated using noise.

<table>
  <!-- <thead>
    <tr>
      <th>visualization</th>
      <th>3D model</th>
      <th>3D model broken</th>
    </tr>
  </thead> -->
  <tbody>
    <tr>
      <td><img width="794" height="598" alt="Image" src="https://github.com/user-attachments/assets/978644c6-ec13-4a93-acef-594c2627fe2d" /></td>
      <td><video src="https://github.com/user-attachments/assets/f2a3e508-5745-4e02-87ef-80a71faf7de9"></video></td> 
    </tr>
  </tbody>  
</table>

# Requirements

* Panda3D 1.10.16
* Cython 3.2.3
* numpy 2.2.6
* scipy 1.16.2
* shapely 2.1.2
* matplotlib 3.10.7
* opencv-contrib-python 4.11.0.86
* opencv-python 4.11.0.86

# Environment

* Python 3.13
* Windows11

# Usage

### Clone this repository.

```
git clone --recursive https://github.com/taKana671/VoronoiCity3.git
```

### Build cython code.

If you do not build the Cython code, Python code will be used to generate noise. <br>
Python code takes longer to generate noise than Cython code.

```
cd VoronoiCity3
python setup.py build_ext --inplace
```

If the error like "ModuleNotFoundError: No module named ‘distutils’" occurs, install the setuptools.

```
pip install setuptools
```

### Run voronoi_city_3.py

```
python voronoi_city_3.py
```

#### Key control

<table>
    <tr>
      <th>key</th>
      <th>description</th>
    </tr>
    <tr>
      <th>Esc</th>
      <th align="left">Close the screen.</th>
    </tr>
    <tr>
      <th>t</th>
      <th align="left">Toggles physical object display on and off.</th>
    </tr>
    <tr>
      <th>w</th>
      <th align="left">Toggles wireframe display on and off.</th>
    </tr>
    <tr>
      <th>v</th>
      <th align="left">Switch between sky view mode and moving view mode.</th>
    </tr>
</table>

In `skyview mode`, you can view the city from above and rotate the entire city by dragging the mouse. 
`moving view mode` allows you to move around the city by keystrokes below.
<table>
    <tr>
      <th>key</th>
      <th>description</th>
    </tr>
    <tr align="left">
      <th>up arrow</th>
      <th>Move forward.</th>
    </tr>
    <tr align="left">
      <th>left arrow</th>
      <th>Turn left.</th>
    </tr>
    <tr align="left">
      <th>right arrow</th>
      <th>Turn right.</th>
    </tr>
    <tr align="left">
      <th>down arrow</th>
      <th>Move backward.</th>
    </tr>
    <tr align="left">
      <th>u</th>
      <th>Go up.</th>
    </tr>
    <tr align="left">
      <th>d</th>
      <th>Go down.</th>
    </tr>
</table>


