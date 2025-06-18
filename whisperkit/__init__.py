import warnings

# jaxtyping/_array_types.py:134: TracerWarning: Converting a tensor to a Python boolean
# might cause the trace to be incorrect. We can't record the data flow of Python values,
# so this value will be treated as a constant in the future. This means that the trace
# might not generalize to other inputs!
warnings.filterwarnings("ignore", module="jaxtyping")
