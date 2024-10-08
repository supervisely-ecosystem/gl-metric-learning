<div align="center" markdown>
<img src="https://i.imgur.com/XLxFpq8.jpg"/>


# Embeddings Calculator

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Demo-Video">Demo Video</a> •
  <a href="#Results">Results</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/gl-metric-learning/supervisely/calculator)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/gl-metric-learning)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/gl-metric-learning/supervisely/calculator.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/gl-metric-learning/supervisely/calculator.png)](https://supervise.ly)

</div>

# Overview

Application calculates **reference embeddings** for the selected Images Project

Application key points:

- Every image in project must have annotation with **item_tag**
- Calculated embeddings can be used in AI Recommendations
- Store embeddings in Team Files

# How to Run

### 1. Add [Embeddings Calculator](https://ecosystem.supervise.ly/apps/gl-metric-learning/supervisely/calculator) to your team from Ecosystem  
<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/gl-metric-learning/supervisely/calculator" src="https://i.imgur.com/8SY4Rxc.png" width="350px" style='padding-bottom: 20px'/>  

### 2. Run app from the context menu of **Images Project**:
ℹ️ You can use [CSV To Images Project](https://ecosystem.supervise.ly/apps/import-csv-catalog) application to get Images Project in suitable format
<img src="https://i.imgur.com/piXysyf.png" width="100%"/>

### 3. Select served session in modal window
<img src="https://i.imgur.com/sqeuh4D.png" width="100%"/>


# Demo Video
<a data-key="sly-embeded-video-link" href="https://youtu.be/xoIzIvvA2b0" data-video-code="xoIzIvvA2b0">
    <img src="https://i.imgur.com/A4bu2Yk.png" alt="SLY_EMBEDED_VIDEO_LINK"  width="70%">
</a>


# Results

After running the application, you will be redirected to the `Tasks` page. Once application processing has finished, your calculations will become available. Click on the file name to proceed to it.

<img src="https://i.imgur.com/4fKEJrn.png"/>
