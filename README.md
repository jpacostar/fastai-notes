# Practical Deep Learning For Coders
# Notes for non command-line fluent speakers

This are my notes while following the [fast.ai](http://course.fast.ai/index.html) _Practical Deep Learning For Coders_ course.  While I have experience with statistics and data analysis in R and Python, starting the course, I had very little practice with Linux systems and had trouble getting basic things done on the terminal (like copying files, making directories, etc.).  I hope this post can help others who are eager to learn deep learning hands-on, but, like me, get stuck before the learning starts!

# First lessons: dogs, cats and more dogs

I started the course towards the end of may 2018.  Right at the end of lesson 1, and eager to try by myself to the dogs-vs-cats classifier, I registered in Paperspace and tried to create a new machine with the fast.ai public template.  However, a message appeared stating that _This instance type has not been enabled in your account yet_.  I filled the request and, three weeks later, haven't heard back.  I also tried the free hour in Crestle, but decided to follow the recomendations towards the end of lesson 2 and try Amazon Web Services (AWS). 

I followed all the steps in the [lesson 2 video](https://youtu.be/JNxcznsrRb8?t=1h57m19s), but when I ran the notebook on an instance of the fast.ai AMI, `torch.cuda.is_available()` returned `False`.  Then, I found this [post](http://forums.fast.ai/t/torch-cuda-is-available-returns-false/16721/), in which the author finally decides to use the Amazon Deep Learning AMI (see [here](https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/) for the basic part of the setup).

## AWS

For setting up an AWS server, see the video of [lesson 2](http://course.fast.ai/lessons/lesson2.html) at 2:00:00.

1. Go to EC2 > launch instance > Community AMIs > search for `fastai`.
1. From a windows computer, save the private key (.pem file you downloaded from AWS) at C:\cygwin64\home\<username>\\.ssh.
1. Select **fastai-part1v2-p2** (prices [here](https://aws.amazon.com/ec2/pricing/)).
1. Filter by **GPU Compute** and select the **p2.xlarge**. Click on **Review and Launch**.
1. Select key pair from first step. Click **Launch**.

To login:

1. Copy IP address from the **Instances** dashboard.
1. From the Terminal: `ssh ubuntu@<IP ADDRESS> -L8888:localhost:8888`.  From Mac: `ssh -i <key file> ubuntu@<IP ADRESS> -L8888:localhost:8888`

To start learning:

1. `cd fastai`
1. `git pull`
1. `conda env update` (once a month)
1. `jupyter notebook`

When finished:

1. At **Instances**: right-click > Instance State > **Stop**.

To send files to aws instance:

* `scp <file to send> ubuntu@<IP ADDRESS>:<dir to send file to>`.

To get files from aws instance:

* `scp ...`
* From notebook: `FileLink(<path>)`

Also, see:

* http://www.fast.ai/2017/11/16/what-you-need/
* http://course.fast.ai/lessons/aws.html
* http://forums.fast.ai/t/torch-cuda-is-available-returns-false/16721/8

To download the models' pretrained weights:

See http://forums.fast.ai/t/dog-breed-challenge-precompute-error/10988/7.

* `wget http://files.fast.ai/models/weights.tgz`
* tar -xvzf weights.tgz


### To configure Amazon deep learning AMI

* https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/
* https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/TroubleshootingInstancesConnecting.html#TroubleshootingInstancesConnectingMindTerm

**Note:** If there is a local jupyter session open, make sure it is not running on port 8888.

```
chmod 600 <key_pair>.pem
ssh -L localhost:8888:localhost:8888 -i <your .pem file name> ubuntu@<Your instance DNS>

# from home (~)
    conda update -n base conda
    pip install --upgrade pip

    git clone https://github.com/jpacostar/fastai-notes.git
    git clone https://github.com/fastai/fastai.git

# from ~/fastai:
    conda env update
    source activate fastai
    pip install environment_kernels

# from ~/fastai/fastai:
    wget http://files.fast.ai/models/weights.tgz
    tar -xvzf weights.tgz
    rm weights.tgz

# from ~/home (~)
    mkdir data
    ln -s ~/data ln fastai-notes/
    ln -s ~/fastai/courses/dl1/planet.py ln ~/fastai-notes/
    
# from ~/fastai-notes
    ln -sf ~/fastai/fastai
    
# from ~/data
    mkdir cifar10
    wget ...
    tar -xvzf cifar10.tgz
    ...
    cd train && find . | grep -o [a-z]*.png | sort -u && cd .
    
    # See http://forums.fast.ai/t/not-a-directory-error-in-cifar10-exercise/13401/5
    mkdir train_ test_
    cd train_
    mkdir airplane automobile bird cat deer dog frog horse ship truck
    cd ..
    function copytrain { for arg in $@; do cp $(find train -name '*'$arg'.png') train_/$arg/; done; };
    copytrain $(ls train_ | grep -o "[a-z]*")
```


## Paperspace

To connect:

* `ssh paperspace@<public ip> -L localhost:8888:localhost:8888`

## Kaggle

See the official [kaggle API](https://github.com/Kaggle/kaggle-api) or the [kaggle-cli](https://github.com/floydwch/kaggle-cli).

With the Kaggle API

1. Download Kaggle credentials from account page into `~/.kaggle/kaggle.json`.
1. `kaggle competitions download -c <competition name>`

With the `kaggle-cli`:

* `kg download -u <username/email> -p <password> -c <competition name>`

```
# from ~/data
    mkdir dog-breed-identification
    cd dog-breed-identification
    kg download -u <...> -p <...> -c dog-breed-identification 
    unzip ...

    mkdir dogs-vs-cats
    cd dogs-vs-cats
    kg download -u <...> -p <...> -c dogs-vs-cats
    unzip ...
    
    cd train
    mkdir cats
    mv cat.* cats/
    mkdir dogs
    mv dog.* dogs/
    
    mkdir valid
    mkdir valid/dogs
    mkdir valid/cats
    mv train/dogs/dog.4* valid/dogs/
    mv train/cats/cat.6* valid/cats/
    
    
    mkdir planet
    cd planet
    kg config -u <...> -p <...> -c planet-understanding-the-amazon-from-space 
    kg download -f test-jpg-additional.tar.7z
    kg download -f test-jpg.tar.7z
    kg download -f test_v2_file_mapping.csv.zip
    kg download -f train-jpg.tar.7z
    kg download -f train_v2.csv.zip
    kg download -f sample_submission_v2.csv.zip
    
    sudo apt install p7zip-full
    
    7za x test-jpg-additional.tar.7z
    tar xf test-jpg-additional.tar
    ...
    
    
    mkdir aclImdb
    wget http://files.fast.ai/data/aclImdb.tgz
    tar -xvzf aclImdb.tgz
    
```

## Git

To clone the repository:

* `git clone https://github.com/jpacostar/fastai-notes.git`

To add files:

1. `git pull`
1. Add the files to the repository local directory.
1. `git add <files>`
1. `git commit <file> -m <commit notes>`
1. `git push https://github.com/jpacostar/fastai-notes.git master`

## Usefull links

* https://www.lifewire.com/linux-terminal-commands-rock-your-world-2201165. See the `nohup` command.
