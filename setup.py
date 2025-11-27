from setuptools import setup,find_packages
import os
def _read(fn):
    with open(os.path.join(os.path.dirname(__file__),fn),encoding='utf-8')as f:return f.read()
def _reqs():
    try:return[l.strip()for l in open('paddleocr_requirements.txt')if l.strip()and not l.startswith('#')]
    except:return['paddlepaddle>=2.5.0','paddleocr>=3.0.0','numpy>=1.20.0','opencv-python>=4.5.0','pillow>=9.0.0','shapely>=1.8.0','pyclipper>=1.3.0']
setup(
    name='document-ocr',
    version='2.1.0',
    author='Jammy',
    author_email='jammy@example.com',
    description='Document OCR Pipeline based on PaddleOCR',
    long_description=_read('docs/README_EN.md')if os.path.exists('docs/README_EN.md')else'',
    long_description_content_type='text/markdown',
    url='https://github.com/jammy/document-ocr',
    packages=find_packages(where='src'),
    package_dir={'':'src'},
    python_requires='>=3.8',
    install_requires=_reqs(),
    extras_require={'dev':['pytest>=7.0','black','flake8'],'gpu':['paddlepaddle-gpu>=2.5.0']},
    entry_points={'console_scripts':['dococr=cli:main']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    keywords='ocr document text-detection text-recognition paddleocr',
)
