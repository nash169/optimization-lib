#!/usr/bin/env python
# encoding: utf-8

import os
import os.path as osp


from wafbuild.utils import load

VERSION = "1.0.0"
APPNAME = "optimization-lib"

libname = "Optimization"
srcdir = "src/optimization_lib"
blddir = "build"

compiler = "cxx"
required = ["eigen", "ipopt"]
optional = ["nlopt"]


def options(opt):
    # Add build shared library options
    opt.add_option("--shared",
                   action="store_true",
                   help="build shared library")

    # Add build static library options
    opt.add_option("--static",
                   action="store_true",
                   help="build static library")

    # Load library options
    load(opt, compiler, required, optional)

    # Load examples options
    opt.recurse("./src/examples")


def configure(cfg):
    # Load library configurations
    load(cfg, compiler, required, optional)

    # Load examples configurations
    cfg.recurse("./src/examples")


def build(bld):
    # Library name
    bld.get_env()["libname"] = libname

    # Includes
    includes = []
    for root, _, filenames in os.walk(srcdir):
        includes += [osp.join(root, filename)
                     for filename in filenames if filename.endswith(('.hpp', '.h'))]

    # Sources
    sources = []
    for root, _, filenames in os.walk(srcdir):
        sources += [osp.join(root, filename)
                    for filename in filenames if filename.endswith(('.cpp', '.cc'))]

    # Build library
    bld.shlib(
        features="cxx cxxshlib",
        source=sources,
        target=bld.get_env()["libname"],
        includes=srcdir,
        uselib=bld.get_env()["libs"],
    ) if bld.options.shared else bld.stlib(
        features="cxx cxxstlib",
        source=sources,
        target=bld.get_env()["libname"],
        includes=srcdir,
        uselib=bld.get_env()["libs"],
    )

    # Build executables
    bld.recurse("./src/examples")

    # Install headers
    [bld.install_files("${PREFIX}/include/" + osp.dirname(f)[4:], f)
     for f in includes]

    # Install libraries
    bld.install_files("${PREFIX}/lib", blddir + "/lib" + bld.get_env()["libname"] + ".a") if bld.env["lib_type"] == "cxxstlib" else bld.install_files(
        "${PREFIX}/lib", blddir + "/lib" + bld.get_env()["libname"] + "." + bld.env.SUFFIX)
