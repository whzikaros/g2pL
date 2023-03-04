import setuptools

setuptools.setup(
    name='g2pL',# 需要打包的名字,即本模块要发布的名字
    version='0.0.18',#版本
    license='Apache License 2.0',
    description='Polyphone disambiguation for g2p', # 简要描述
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),   #  需要打包的模块
    author='HongZhi Wan', # 作者名
    author_email='whzikaros@gmail.com',   # 作者邮件
    url='https://github.com/whzikaros/g2pL', # 项目地址,一般是代码托管的网站
    python_requires=">=3.6",
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3"
    ],    
    install_requires=['transformers==3.4.0','opencc==1.1.3',"requests==2.28.1"], # 依赖包,如果没有,可以不要
)    