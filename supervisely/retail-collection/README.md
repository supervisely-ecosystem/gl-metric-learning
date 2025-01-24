<div align="center" markdown>
  
<h1 align="center" style="border-bottom: 0"> 🧃 Retail Collection </h1>

  <p align="center"><b>Label images using updatable Reference Database</b></p>

  
<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Launch">Launch</a> •
  <a href="#Map">Map</a> •
  <a href="#Applications">Applications</a> •
  <a href="#Demo-Data">Demo Data</a> •
  <a href="#For-Developers">For Developers</a> •
  <a href="#About-Us">About Us</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/gl-metric-learning/supervisely/retail-collection)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/gl-metric-learning)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/gl-metric-learning/supervisely/retail-collection.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/gl-metric-learning/supervisely/retail-collection.png)](https://supervisely.com)


<p float="left">
  <img src="https://github.com/supervisely-ecosystem/metric-learning-collection/releases/download/v0.0.1/retail-bundle-demo.gif?raw=true" style="width:80%;"/>
</p>
  
</div>

## Overview

This collection allows you to label images classes using predictions of pretrained Metric Learning model and Reference Database.  

All you need to start is:
- .CSV catalog with `image_url` and `item_id` fields
- Agent with `GPU`


<table>
    <tr style="width: 100%">
        <td >
          <img src="https://imgur.com/4fZNO25.png" style=""/>
            <h4 align="center" style="font-family:'Lucida Console', monospace; padding-top: 3px; padding-bottom: 0">assign tags using NN predictions</h4>
        </td>
        <td>
          <img src="https://imgur.com/KRcUqSg.png" style=""/>
            <h4 align="center" style="font-family:'Lucida Console', monospace; padding-top: 3px; padding-bottom: 0">add new items to Reference Database</h4> 
        </td>
    </tr>
    <tr>
        <td>
          <img src="https://imgur.com/VI5mcA1.png" style=""/>
            <h4 align="center" style="font-family:'Lucida Console', monospace; padding-top: 3px">review assigned tags</h4> 
        </td>
        <td>
          <img src="https://imgur.com/rrDFVQP.png" style=""/>
            <h4 align="center" style="font-family:'Lucida Console', monospace; padding-top: 3px">manual search in Reference Database</h4> 
        </td>
    </tr>
    
</table>


## Launch

<img src="https://imgur.com/ejK0mHt.png" style="width:100%;"/>


## Map

<p>This map illustrates how each application in the collection connected to each other</p>
<img src="https://imgur.com/pwPAdqb.png" style="width:100%;"/>


## Applications

- [CSV Products Catalog To Images Project](https://ecosystem.supervisely.com/apps/import-csv-catalog) 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-csv-catalog" src="https://imgur.com/NxeAATL.png" width="350px"/> 

- [Serve Metric Learning](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/gl-metric-learning/supervisely/serve)

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/gl-metric-learning/supervisely/serve" src="https://imgur.com/A3BW6hP.png" width="350px"/> 

- [Embeddings Calculator](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/gl-metric-learning/supervisely/calculator)

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/gl-metric-learning/supervisely/calculator" src="https://imgur.com/QL90cJS.png" width="350px"/>  

- [AI Recommendations](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/gl-metric-learning/supervisely/similarity-calculator)

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/gl-metric-learning/supervisely/similarity-calculator" src="https://imgur.com/WptA30Z.png" width="350px"/> 

- [Metric Learning Labeling Tool](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/gl-metric-learning/supervisely/labeling-tool)

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/gl-metric-learning/supervisely/labeling-tool" src="https://imgur.com/8HQvAuT.png" width="350px"/>  
    
    
## Demo Data

- [.CSV table to import example](https://github.com/supervisely-ecosystem/import-csv-catalog/releases/download/v0.0.1/test_snacks_catalog.csv) — the table that is used in the **first step** of [launch](#Launch) 
- [Grocery store shelves](https://ecosystem.supervisely.com/projects/grocery-store-shelves) - labeled and untagged images for test

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/grocery-store-shelves" src="https://i.imgur.com/Mqqqs4c.png" width="350px"/>


# For Developers

You can use sources from from any application to create your own.

You can also refer to our documentation:

- [How to create Superivsely APP](https://github.com/supervisely-ecosystem/how-to-create-app)
- [Learn SDK Basics with IPython Notebooks](https://sdk.docs.supervisely.com/rst_templates/notebooks/notebooks.html)
- [Complete Python SDK](https://sdk.docs.supervisely.com/sdk_packages.html)

# About us

You can think of [Supervisely](https://supervisely.com/) as an Operating System available via Web Browser to help you solve
Computer Vision tasks. The idea is to unify all the relevant tools that may be needed to make the development process as
smooth and fast as possible.

More concretely, Supervisely includes the following functionality:

- Data labeling for images, videos, 3D point cloud and volumetric medical images (dicom)
- Data visualization and quality control
- State-Of-The-Art Deep Learning models for segmentation, detection, classification and other tasks
- Interactive tools for model performance analysis
- Specialized Deep Learning models to speed up data labeling (aka AI-assisted labeling)
- Synthetic data generation tools
- Instruments to make it easier to collaborate for data scientists, data labelers, domain experts and software engineers

One challenge is to make it possible for everyone to train and apply SOTA Deep Learning models directly from the Web
Browser. To address it, we introduce an open sourced Supervisely Agent. All you need to do is to execute a single
command on your machine with the GPU that installs the Agent. After that, you keep working in the browser and all the
GPU related computations will be performed on the connected machine(s).

- for technical support please leave issues, questions or suggestions in
  our [repo](https://github.com/supervisely-ecosystem/gl-metric-learning). Our team will try to help.
- also we can chat in slack
  channel [![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
- if you are interested in Supervisely Enterprise Edition (EE) please send us
  a [request](https://supervisely.com/enterprise/?demo) or email Yuri Borisov at [sales@supervisely.com](sales@supervisely.com)
