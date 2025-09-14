from models.ResNetMAML import ResNetMAML
import torch.nn.functional as F

class CNNMAML(ResNetMAML):
  def forward(self, x, params=None):
    if not params: params = dict(self.named_parameters())  # uses meta/global params when local params not given
    x = F.conv2d(
      input=x,
      weight=params[f'convs.{0}.weight'],
      bias=params[f'convs.{0}.bias'],
      stride=self.config.stride,
      padding=self.config.padding
    )  # first conv
    for i in range(1, self.convs.__len__()):
      x = F.conv2d(
        input=x,
        weight=params[f'convs.{i}.weight'],
        bias=params[f'convs.{i}.bias'],
        stride=self.config.stride,
        padding=self.config.padding,
      )  # hidden convs
      x = self.act(x)
    x = self.pool(x)
    x = self.flat(x)
    return F.linear(x, weight=params['fc.weight'], bias=params['fc.bias'])
  # forward
# CNNMAML