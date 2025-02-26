------------------
CODE originally made by Paolo Papale, forked from the HersenInstituut/MEIs_mice 
------------------

## MEIs_mice

<img src="/_readme/meis.png" width="100">

This code will allow you to fit ANNs to your 2p data and get tunings and most exciting inputs (MEIs)! This code requires 0 (previous) python knoweldge. However, this will only work on the `MVP-server` and only with `Res` files coming out of Chris vdT.'s 2p pre-processing pipeline! It allows any N-by-N experimental desing!

#### To install the software requirments and run your model, follow these steps:

#### 1. Fork and clone this repo

You need to fork this repo to your GitHub account, and then clone it to your MVP server user folder (e.g. under Documents/GitHub). Forking is needed so you can keep track of your own changes without messing up with everyone else's paths etc.

#### 2. Import the environment in Anaconda Navigator

Open Anaconda Navigator (it's already installed for every MVP user). Then go to the "Environments" tab and look for the "Import" icon at the bottom:

<img src="/_readme/import.png" width="50">

Select `Torch_gpu.yaml` from the repo folder you cloned, confirm and wait. Once loaded, activate the environmnet by clicking on it.

#### 3. Install GPU utils

Then, click on the green 'Play' button and select "Open Terminal":

<img src="/_readme/terminal.png" width="500">

Copy and paste the following command in the Terminal:

`pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cu117`

This is gonna take a while (~1-2 hours), but you just need to do it once! It's gonna ask for confirmation, type 'Y' and wait. Once finished, close the terminal.

#### 4. Check the installation

You're almost there! Go back to Anaconda Navigator, click on the "Home" tab and then launch a JupyterLab session:

<img src="/_readme/jupyter.png" width="500">

Navigate to the folder where you cloned this repo from the menu on the left:

<img src="/_readme/nav.png" width="200">

And then find and double click on `Test_installation.ipynb`. If you're not familiar with JupyterLab, find in the menu and click on "Run">"Run All Cells". If no warnings are issued and the second cell returns "True", everything went well and you're ready to go! Otherwise, just get in touch!

#### 5. Run the model and get your tunings and MEIs!

Everything is controlled via `NPC_mice.ipynb`. Just change the constants in the first cell to match the path to your datasets and the values you whish to test for orientation and size tuning. Then "Run">"Run All Cells". It can take a long time to pre-process the data and then run the models.

#### N.B.

The code is not be publicly shared in this form, because I included the "lucent" library here for simiplicity, but that'd go under a different license. If you use this code in a publication, cite:

*Seignette, Koen, et al. "Experience Shapes Chandelier Cell Function and Structure in the Visual Cortex." eLife 12 (2023).*

You can use this reference also to get more insight on the pipeline or you can copy/paste the methods from there. 
If you need something specific, and you don't get it to work by yourself, just get in touch!
