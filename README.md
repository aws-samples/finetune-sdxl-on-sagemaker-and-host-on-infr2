# Fine-tune And host SDXL models cost-effectively with AWS Inferentia2

## Overview

This repository offers a comprehensive example of how to fine-tune the state-of-the-art Stable Diffusion XL (SDXL) text-to-image generation model using Dreambooth and LoRA techniques on Amazon SageMaker. It demonstrates an efficient approach to personalize the SDXL model with just a few input images, enabling the generation of highly customized and domain-specific images tailored to individual requirements. Additionally, the blog covers the process of compiling and deploying the fine-tuned SDXL model on AWS Inferentia2 (Inf2) instances, leveraging the exceptional performance and cost-efficiency of these specialized accelerators for generative AI workloads.

### Efficient Fine-tuning SDXL using Dreambooth and LoRA
Dreambooth allows you to personalize the model by embedding a subject into its output domain using a unique identifier, effectively expanding its language-vision dictionary. This process leverages a technique called prior preservation, which retains the model's existing knowledge about the subject class (e.g., humans) while incorporating new information from the provided subject images. LoRA is an efficient fine-tuning method that attaches small adapter networks to specific layers of the pre-trained model, freezing most of its weights. By combining these techniques, you could generate a personalized model while tuning an order-of-magnitude fewer parameters, resulting in faster fine-tuning times and optimized storage requirements.

### Prepared fine-tuned Model for Inf2

Prepared fine-tuned Model for Inf2 After the model is fine-tuned, you will compile and host the fine-tuned SDXL on Inf2 instances using Torch Neuron. By doing this, you can benefit from the exceptional performance and cost-efficiency offered by these specialized accelerators while taking advantage of the seamless integration with popular deep learning frameworks like TensorFlow and PyTorch. To learn more, please visit our  AWS Inf2 and Neuron product pages.

## Prerequisites

Before you get started let's review list of services and instance types required to run the sample notebooks provided in this repository.

* Basic understanding of stable diffusion models. Please refer to this [Create high-quality images with Stable Diffusion models and deploy them cost-efficiently with Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/create-high-quality-images-with-stable-diffusion-models-and-deploy-them-cost-efficiently-with-amazon-sagemaker/) for more information.
* General knowledge about foundation models and how fine-tuning brings value, read more on [Fine-tune a foundation model](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-fine-tuning.html).
* An AWS account. Ensure your AWS identity has the requisite permissions, including the ability to create SageMaker Resources (Domain, Model, and Endpoints) and Amazon S3 access to upload model artifacts. Alternatively, you can attach the AmazonSageMakerFullAccess managed policy to your IAM User or Role.
* This notebook is tested using the default `Python 3 kernel` on SageMaker Studio. A GPU instance such as `ml.g5.2xlarge` is recommended. Please refer to the documentation on [Setting up a Domain for Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html).
* For compiling the fine-tuned model, an `inf2.8xlarge` or larger EC2 instance with Jupyter Lab installed is required. follow [Set up a Jupyter Notebook Server](https://docs.aws.amazon.com/dlami/latest/devguide/setup-jupyter.html) for step-by-step instruction.

By following these prerequisites, you will have the necessary knowledge and AWS resources to run the sample notebooks and work with stable diffusion models and foundation models on Amazon SageMaker.

## Steps to follow

#### Fine-tune sdxl

1. Clone this repo.

```shell
git clone https://github.com/aws-samples/finetune-sdxl-on-sagemaker-and-host-on-infr2.git
```

2. Upload 10-12 selfie images in `fine-tune-sdxl/data/` directory.

[Sample Images]

3. Run the notebook `sdxl_fine_tune.ipynb` in fine-tune-sdxl folder to fine-tune and test SDXL model with your images.

#### Compile on Inferentia

1. Create an `inf2.8xlarge` inferentia EC2 instance, install and run jupiter notebook follow the instruction in Prerequisites section.
2. Clone this repo onto the instance.

```shell
git clone https://github.com/aws-samples/finetune-sdxl-on-sagemaker-and-host-on-infr2.git
```

3. Execute the scripts  `compile-on-inferentia/install-drivers.sh` and `compile-on-inferentia/install-pytorch-neuron.sh` to install Neuron runtime, driver, and etc.

```shell
sh compile-on-inferentia/install-drivers.sh

sh compile-on-inferentia/install-pytorch-neuron.sh
```

4. Copy the model weights generated in `fine-tune-sdxl/finetune_sttirum` directory to `compile-on-inferentia/lora` directory
5. Run the notebook `compile-on-inferentia/stable-diffusion-xl-lora_inference.ipynb` in `compile-on-inferentia` folder to compile and then test SDXL model with LoRA adapter on inferentia2 


## Example avatar images and sample prompts

[Avatar Images]


## Cleanup

To avoid incurring AWS charges after you are done testing the guidance, make sure you delete the following resources:

* Amazon SageMaker Studio Domain
* Amazon EC2 Inf2 EC2 instance


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
