# Markitdown
# https://github.com/microsoft/markitdown

from markitdown import MarkItDown

md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
result = md.convert("ICA-Supermarket-Medis-2025-09-27.pdf")
print(result) # result in the file ICA-Supermarket-Medis-2025-09-27.md