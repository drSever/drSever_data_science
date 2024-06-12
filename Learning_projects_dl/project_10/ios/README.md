# iOS-application
[Link](https://youtu.be/ypb2P3RHKqA) to the video of the prototype work.

The **original repository** on which the solution is based https://github.com/tensorflow/examples

# Style Transfer iOS sample (original)

![Screenshot](https://storage.googleapis.com/download.tensorflow.org/models/tflite/arbitrary_style_transfer/architecture.png)

**! Note:** our solution uses InceptionV3 instead of MobilNetV2

## Requirements

*   Xcode 11 or higher (installed on a Mac machine)
*   An iOS Simulator or device running iOS 10 or above
*   Xcode command-line tools (run `xcode-select --install`)
*   CocoaPods (run `sudo gem install cocoapods`)

**! Note:** If `sudo gem install cocoapods` installation fails, uninstall the installed packages and install via [brew](https://brew.sh): `brew install cocoapods`.

## Tested on

MacBook M1 pro 16Gb, MacOS Sonoma 14.5, Xcode 15.4

## Build and run

1.  Clone the repository to your computer to get the
    demo application:     
    `git clone https://github.com/drSever/drSever_data_science` 

    or clone original repository:    
    `git clone https://github.com/tensorflow/examples`    
    in this case, the source code will be located in    
    `lite/examples/style_transfer/ios`

    You can also clone the repository when you start Xcode
2.  Install the pod to generate the workspace file:     
    `cd <path....>/ios`    
    `pod install`   
    At the end of this step you should have a directory called
    `StyleTransfer.xcworkspace`.
3.  Open the project in Xcode and run it.
4.  The original default models will appear in the   
    `/StyleTransfer/model` folder.
5.  Replace the model data with your own models, for example mine, which are in the 
    `/my_models` folder (If you use your own models for replacement, rename them accordingly).     
    You can also adjust the *download_tflite_models.sh* file by changing the source of model downloads.
6.  Open the project in Xcode and run it again.
7.  Enjoy!
