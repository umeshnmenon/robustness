import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='model_robustness',
      version='0.1',
      description='Calculates Robustness Index using bootstrapped samples',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Umesh Narayana Menon',
      packages= setuptools.find_packages(), #['harness'],
      keywords = ['robustness', 'robust'],
      zip_safe=False
)