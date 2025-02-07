import streamlit as st
import time
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.cuda
from torchinfo import summary


def home():
    st.title("🚀 Deep Learning Training Analyzer")

    st.write(
        "### Welcome to the ultimate tool for analyzing your deep learning model's memory and computational requirements! 📊💡")

    st.markdown(
        """
        🔍 **What does this app do?**
        - It helps estimate memory usage, compute gradients, optimizer parameters, and layer activations.
        - It predicts model loss and outputs while calculating total training size requirements.
        - It provides insights into how long training will take on a suggested GPU. 🎯

        🏆 **How does it work?**
        - Use **Pretrained Models** to analyze common architectures with batch size & data points.
        - Use **Custom Models** to upload your own model code and get detailed training insights.

        🚦 **Let's get started! Choose an option from the sidebar to begin.**
        """
    )

    # Add a simple loading animation
    with st.spinner("Loading app features..."):
        time.sleep(1)
    st.success("All set! Choose an option from the sidebar to proceed. ✅")

    st.info('''Today, large models with billions of parameters are trained with many GPUs across several machines in parallel. 
        Even a single H100 GPU with 80 GB of VRAM (one of the biggest today) is not enough to train just a 30B parameter model (even with batch size 1 and 16-bit precision). 
        The memory consumption for training is generally made up of

        1. the model parameters,
        2. the layer activations (forward),
        3. the gradients (backward),
        4. the optimizer states (e.g., Adam has two additional exponential averages per parameter) and
        5. model outputs and loss.

        When the sum of these memory components exceed the VRAM of a single GPU, regular data-parallel training (DDP) can no longer be employed. 
        To alleviate this limitation, we need to introduce Model Parallelism.''', icon="ℹ️")

    st.divider()

    st.markdown("""| Type                                   | vRAM/GPU     | vCPUs  | RAM | Storage      | PRICE/GPU/HR*     |
|----------------------------------------|---------|---------|------|---------|------------|
| On-demand 1x NVIDIA GH200              | 96 GB   | 64      | 432 GiB  | 4 TiB SSD   | $1.49 / GPU / hr |
| On-demand 8x NVIDIA H100 SXM           | 80 GB   | 208     | 1800 GiB | 22 TiB SSD  | $2.99 / GPU / hr |
| Reserved 8x NVIDIA H100 SXM            | 80 GB   | 208     | 1800 GiB | 22 TiB SSD  | CONTACT SALES    |
| On-demand 4x NVIDIA H100 SXM           | 80 GB   | 104     | 900 GiB  | 11 TiB SSD  | $3.09 / GPU / hr |
| On-demand 2x NVIDIA H100 SXM           | 80 GB   | 52      | 450 GiB  | 5.5 TiB SSD | $3.19 / GPU / hr |
| On-demand 1x NVIDIA H100 SXM           | 80 GB   | 26      | 225 GiB  | 2.75 TiB SSD| $3.29 / GPU / hr |
| On-demand 1x NVIDIA H100 PCIe          | 80 GB   | 26      | 225 GiB  | 1 TiB SSD   | $2.49 / GPU / hr |
| On-demand 8x NVIDIA A100 SXM           | 80 GB   | 240     | 1800 GiB | 19.5 TiB SSD| $1.79 / GPU / hr |
| On-demand 8x NVIDIA A100 SXM           | 40 GB   | 124     | 1800 GiB | 5.8 TiB SSD | $1.29 / GPU / hr |
| On-demand 1x NVIDIA A100 SXM           | 40 GB   | 30      | 220 GiB  | 512 GiB SSD | $1.29 / GPU / hr |
| On-demand 4x NVIDIA A100 PCIe          | 40 GB   | 120     | 900 GiB  | 1 TiB SSD   | $1.29 / GPU / hr |
| On-demand 2x NVIDIA A100 PCIe          | 40 GB   | 60      | 450 GiB  | 1 TiB SSD   | $1.29 / GPU / hr |
| On-demand 1x NVIDIA A100 PCIe          | 40 GB   | 30      | 225 GiB  | 512 GiB SSD | $1.29 / GPU / hr |
| On-demand 1x NVIDIA A10                | 24 GB   | 30      | 226 GiB  | 1.3 TiB SSD | $0.75 / GPU / hr |
| On-demand 4x NVIDIA A6000              | 48 GB   | 56      | 400 GiB  | 1 TiB SSD   | $0.80 / GPU / hr |
| On-demand 2x NVIDIA A6000              | 48 GB   | 28      | 200 GiB  | 1 TiB SSD   | $0.80 / GPU / hr |
| On-demand 1x NVIDIA A6000              | 48 GB   | 14      | 100 GiB  | 512 GiB SSD | $0.80 / GPU / hr |
| On-demand 8x NVIDIA Tesla V100         | 16 GB   | 88      | 448 GiB  | 5.8 TiB SSD | $0.55 / GPU / hr |
| On-demand 1x NVIDIA Quadro RTX 6000    | 24 GB   | 14      | 46 GiB   | 512 GiB SSD | $0.50 / GPU / hr |
""")
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def Pretrained():
    st.title("Pretrained Models")
    st.info('This is a purely informational message', icon="ℹ️")

    model_mapping = {
        "ResNet18": models.resnet18,
        "ResNet34": models.resnet34,
        "ResNet50": models.resnet50,
        "ResNet101": models.resnet101,
        "ResNet152": models.resnet152,
        "EfficientNet-B0": models.efficientnet_b0,
        "EfficientNet-B1": models.efficientnet_b1,
        "EfficientNet-B2": models.efficientnet_b2,
        "EfficientNet-B3": models.efficientnet_b3,
        "EfficientNet-B4": models.efficientnet_b4,
        "EfficientNet-B5": models.efficientnet_b5,
        "EfficientNet-B6": models.efficientnet_b6,
        "EfficientNet-B7": models.efficientnet_b7,
        "VGG11": models.vgg11,
        "VGG13": models.vgg13,
        "VGG16": models.vgg16,
        "VGG19": models.vgg19,
        "DenseNet121": models.densenet121,
        "DenseNet161": models.densenet161,
        "DenseNet169": models.densenet169,
        "DenseNet201": models.densenet201,
        "MobileNetV2": models.mobilenet_v2,
        "MobileNetV3 Small": models.mobilenet_v3_small,
        "MobileNetV3 Large": models.mobilenet_v3_large,
        "SqueezeNet1.0": models.squeezenet1_0,
        "SqueezeNet1.1": models.squeezenet1_1,
        "AlexNet": models.alexnet,
        "GoogLeNet": models.googlenet,
        "ShuffleNetV2": models.shufflenet_v2_x1_0,
        "MNASNet": models.mnasnet1_0,
    }

    # Model List for Dropdown
    model_list = list(model_mapping.keys())

    with st.container(height=105,border=True):
        # Create two columns for layout organization
        col1, col2, col3 = st.columns(3)
        with col1:
            pretrained_model = st.selectbox("Select Pretrained Model", options=model_list)
        with col2:
            batch_size = st.number_input("Enter the Batch size",min_value=1,max_value=1024,value=32,step=1)
        with col3:
            data_size = st.number_input("Enter the Size of data",min_value=1,value=10000,step=500)

    # Load the selected model
    selected_model = model_mapping.get(pretrained_model, None)

    # ---------- Calculate Model Parameter Size ----------
    def get_size_in_bytes(tensor):
        return tensor.numel() * tensor.element_size()

    # ---------- Convert Sizes to Human-Readable Format ----------
    def convert_size(size_in_bytes):
        if size_in_bytes < 1024:
            return f"{size_in_bytes} B"
        elif size_in_bytes < 1024 ** 2:
            return f"{size_in_bytes / 1024:.2f} KB"
        elif size_in_bytes < 1024 ** 3:
            return f"{size_in_bytes / 1024 ** 2:.2f} MB"
        else:
            return f"{size_in_bytes / 1024 ** 3:.2f} GB"

    if st.button("Suggest",help="calculate the VRAM needed to your model and which GPU is best"):
        with st.status("Calculating all Those training factors", expanded=True) as status:
            st.write("Generate Model Summary")
            model = selected_model()
            param_size = sum(get_size_in_bytes(p) for p in model.parameters())
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            x = torch.randn(batch_size, 3, 224, 224).to(device)
            # ---------- Model Summary ------------------------------------------------------
            used_model = str(model)
            model_input_shape = str(x.shape)
            model_summaries = "\n📜 Model Summary:\n"
            model_summary = str(summary(model, input_size=x.shape))
            model_summary_text = f"{used_model}\n {model_input_shape} \n{model_summaries}{model_summary}\n"

            st.write("Run the model")
            # -------------- Model Run -----------------------------------------------------
            output_size = model(x).shape  # Get output size
            y_true = torch.randn(output_size).to(device)  # Create random target
            # Define loss function
            loss_fn = nn.MSELoss()
            # Define optimizer (Adam)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            # --- Forward Pass ---
            y_pred = model(x)  # Forward pass
            loss = loss_fn(y_pred, y_true)  # Compute loss
            # --- Backward Pass ---
            loss.backward()
            # --- Optimizer Step ---
            optimizer.step()

            st.write("Model Parameter Info")
            # --- Model Parameter Info ------------------------------------------------------
            model_parameter = "\n🚀 Model Parameter Details:"
            total_params = sum(p.numel() for p in model.parameters())
            param_size_shape = ""
            for name, param in model.named_parameters():
                param_size_shape += f"{name}: {param.shape}, Size: {param.numel()}" + "\n"
            total_params1 = f"🔹 Total Parameters: {total_params}\n"
            param_size = f"🔹 Total Parameter Size: {convert_size(sum(get_size_in_bytes(p) for p in model.parameters()))} \n"
            param_size_text = f"{model_parameter}\n{param_size_shape}{total_params1}{param_size}"

            st.write("Layer Activation")
            # --- Layer Activations (Forward Pass) --------------------------------------------
            activations = {}
            def hook_fn(name):
                def hook(module, input, output):
                    activations[name] = output.detach()
                return hook
            # Register hooks to capture activations
            for name, layer in model.named_children():
                layer.register_forward_hook(hook_fn(name))
            # Run forward pass to collect activations
            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)
            # Compute activation sizes
            activation_size = sum(get_size_in_bytes(a) for a in activations.values())
            # --- Print Activation Details ---
            Layer_activation = "\n⚡ Layer Activations:"
            activation_shape = 0
            Layer_activations = ""
            for name, act_tensor in activations.items():
                activation_shape = activation_shape + act_tensor.numel()
                Layer_activations += f"{name}: Shape={list(act_tensor.shape)}, Size={get_size_in_bytes(act_tensor)}" + "\n"
            Activation_shape = f"\n🔹 Activation Shape: {activation_shape}"
            Activation_size = f"\n🔹 Total Activation Size: {convert_size(activation_size)}"
            activation_text = f"{Layer_activation}\n{Layer_activations}{Activation_shape}{Activation_size}"

            st.write("Gradient Information (Backward Pass)")
            # --- Gradient Information (Backward Pass) ------------------------------------------
            total_gradients = sum(p.numel() for p in model.parameters() if p.grad is not None)
            total_gradients_size = sum(get_size_in_bytes(p.grad) for p in model.parameters() if p.grad is not None)
            model_gradient = "\n🔄 Gradients (Backward Pass):"
            gradient_content = ""
            for name, param in model.named_parameters():
                gradient_content += f"{name} Gradient Shape: {param.grad.shape if param.grad is not None else None}" + "\n"
            Total_gradient = f"🔹 Total Gradients: {total_gradients}"
            Total_gradient_size = f"🔹 Total Gradient Size: {convert_size(total_gradients_size)}\n"
            gradient_text = f"{model_gradient}\n{gradient_content}{Total_gradient}{Total_gradient_size}"

            st.write("Optimizer State Info ")
            # --- Optimizer State Info ---------------------------------------------------------
            optimizer_state = "\n🛠️ Optimizer State:"
            exp_avg = 0
            exp_avg_sq = 0
            exp_avg_size = 0
            exp_avg_sq_size = 0
            for i in optimizer.state_dict()['state'].keys():
                exp_avg = (exp_avg + optimizer.state_dict()['state'][i]['exp_avg'].numel()) + (exp_avg_sq + optimizer.state_dict()['state'][i]['exp_avg_sq'].numel())
                exp_avg_size = exp_avg_size + get_size_in_bytes(optimizer.state_dict()['state'][i]['exp_avg'])
                exp_avg_sq_size = exp_avg_sq_size + get_size_in_bytes(optimizer.state_dict()['state'][i]['exp_avg_sq'])
            total_optimizer_state = f"🔹 Total Optimizer State: {exp_avg}"
            total_optimizer_size = f"🔹 Total Optimizer State Size: {convert_size(exp_avg_size + exp_avg_sq_size)} "

            optimizer_text = f"{optimizer_state}\n{total_optimizer_state}{total_optimizer_size}"

            st.write("Model Output Loss")
            # ----------- Model Outputs and Loss ----------------------------------------------------
            output_loss = "\n📈📉 Model Output and Loss"
            info = "It is already calculated in Layers activation"
            model_output_shape = f"🔹 model output shape if the batch size is {batch_size} : {act_tensor.numel()}"
            model_output_size = f"🔹 model output Size if the batch size is {batch_size} : {convert_size(get_size_in_bytes(act_tensor))}"
            model_output_text = f"{output_loss}\n{ info}\n{model_output_shape}\n{model_output_size}"
            status.update(
                label="Process Complete!", state="complete", expanded=False
            )

        # Final Summary
        final_summary = f"{model_summary_text}\n{param_size_text}\n{activation_text}\n{gradient_text}\n{optimizer_text}\n{model_output_text}"
        model_config = f"{model_parameter}\n{total_params1}{param_size}\n{Layer_activation}{Activation_shape}{Activation_size}\n{model_gradient}\n{Total_gradient}\n{Total_gradient_size}\n{optimizer_state}\n{total_optimizer_state}\n{total_optimizer_size}\n\n{output_loss}\n{info}\n{model_output_shape}\n{model_output_size}"
        text_col1, text_col2 = st.columns(2)
        with text_col1:
            with st.container(height=800):
                st.text_area("Model Summary", model_summary, height=700)
        with text_col2:
            with st.container(height=800):
                st.text_area("Model Configuration", model_config, height=700)

        st.download_button(
            label="Download Model Summary",
            data=final_summary,
            file_name=f"model_summary.txt",
            mime="text/plain"
        )
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def Custom():
    import streamlit as st
    import torch
    import torch.nn as nn
    import torchinfo

    # Streamlit UI
    st.title("🧠 PyTorch Model Summary Viewer 🔍")

# Dictionary to manage pages
pages = {
    "Home": home,
    "Pretrained": Pretrained,
    "Custom": Custom,
}

# Sidebar navigation
# Set up the Streamlit page configuration
st.set_page_config("Diagnosis", layout='wide')
# st.divider()
with st.sidebar:
    st.title("Navigation")
    selection = st.radio("Go to", list(pages.keys()))


# Display the selected page
pages[selection]()
