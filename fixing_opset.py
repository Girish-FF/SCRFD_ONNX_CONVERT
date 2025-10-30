import onnx

# Load your existing model
model = onnx.load("scrfd_with_post.onnx")

# Add opset import for the custom domain
custom_opset = onnx.helper.make_opsetid("ai.onnx.contrib", 1)

# Check if the domain already exists and update, or add new
found = False
for opset in model.opset_import:
    if opset.domain == "ai.onnx.contrib":
        opset.version = 1
        found = True
        break

if not found:
    model.opset_import.append(custom_opset)

# Save the updated model
onnx.save(model, "scrfd_with_post_fixed.onnx")
