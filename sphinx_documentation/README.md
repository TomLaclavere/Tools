The tutorial to build documentation using Sphinx can be found following: https://github.com/sfarrens/ecole-euclid-2023?tab=readme-ov-file

The commands used are the following.

# Build Documentation
In the main branch, generate your documentation using Sphinx : 

```bash
sphinx-quickstart docs/
sphinx-apidoc -Mfeo docs/source module_path
```

If you want to verify your documentation, you can build it with :

```bash
sphinx-build docs/source docs/build
firefox docs/build/index.html
```

# Configure gh-pages

The next step is to build and configure a gh-pages. 

## Build gh-pages

You can build this specific branch following :

```bash
git checkout --orphan gh-pages
git reset --hard
echo "gh-pages branch for GitHub Pages" > README.md
git add README.md
git commit -m "Initialize gh-pages branch"
git push origin gh-pages
```

## Activate GitHub pages

To activate this functionality in your GitHub repository, you have to go into settings and click on Pages. In the "Build and Deployment" section, select "Deploy from a branch" in Source, and select the gh-pages in the Branch section.
Validate by clicking on save.

# Add GitHub Actions Workflow

Come back into your main branch and create the yaml file to configure the Continuous Deployment in the following path :

```bash
.github/workflows/cd.yml
```

I put an example of this file on my repository, you can use the same lines but do not forget to modify the different paths written.

# Deployment

After committing and pushing any change in the main branch, you will see in the Actions section the execution of the different commands we wrote. Please feel free to check if everything works as expected.
If everything runs well, you can find your documentation following the https address : 

```bash
<username>.github.io/<repository_name>/index.html
```








