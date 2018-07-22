import numpy as np

class DftLayer (Function):

    @staticmethod
    def forward (ctx, input, weight):
        """
        :param input: (array (batch_size, input_size))
        :param weight: (array (max(input_size, output_size),))
        """

        ctx.save_for_backward(input, weight)
        output =
