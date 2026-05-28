import timm
from torchinfo import summary


model = timm.create_model('tf_mobilenetv3_small_100.in1k', pretrained=True, num_classes=8)
summary(model, input_size=(1, 3, 224, 224), col_names=('input_size', 'output_size', 'num_params', 'mult_adds', 'trainable'))
