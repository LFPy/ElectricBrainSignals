import os
import sys
from os.path import join
from glob import glob
import posixpath
import numpy as np
import neuron
import LFPy
from brainsignals import cell_models

h = neuron.h
cell_models_folder = os.path.abspath(cell_models.__path__[0])
hay_folder = join(cell_models_folder, "L5bPCmodelsEH")
hallermann_folder = join(cell_models_folder, "HallermannEtAl2012")
allen_folder = join(cell_models_folder, "allen")
bbp_folder = join(cell_models_folder, "bbp_models")
bbp_mod_folder = join(cell_models_folder, "bbp_mod")


def load_mechs_from_folder(mod_folder):
    # TODO: HAS NOT BEEN TESTED ON WINDOWS
    mechs_loaded = neuron.load_mechanisms(mod_folder)
    if not mechs_loaded:
        print("Attempting to compile mod mechanisms.")
        if "win32" in sys.platform:
            warn("no autompile of NMODL (.mod) files on Windows.\n"
                 + "Run mknrndll from NEURON bash in the folder ")
            if not mod_folder in neuron.nrn_dll_loaded:
                neuron.h.nrn_load_dll(join(mod_pth, "nrnmech.dll"))
            neuron.nrn_dll_loaded.append(mod_folder)
        else:
            os.system('''
                      cd {}
                      nrnivmodl
                      cd -
                      '''.format(mod_folder))
            mechs_loaded = neuron.load_mechanisms(mod_folder)
            if not mechs_loaded:
                raise RuntimeError("Could not load mechanisms")


def download_hay_model():

    print("Downloading Hay model")
    if sys.version < '3':
        from urllib2 import urlopen
    else:
        from urllib.request import urlopen
    import ssl
    from warnings import warn
    import zipfile
    #get the model files:
    u = urlopen('https://modeldb.science/download/139653',
                context=ssl._create_unverified_context())
    localFile = open(join(cell_models_folder, 'L5bPCmodelsEH.zip'), 'wb')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile(join(cell_models_folder, 'L5bPCmodelsEH.zip'), 'r')
    myzip.extractall(cell_models_folder)
    myzip.close()

    #compile mod files every time, because of incompatibility with Mainen96 files:
    mod_pth = join(hay_folder, "mod/")

    if "win32" in sys.platform:
        warn("no autompile of NMODL (.mod) files on Windows.\n"
             + "Run mknrndll from NEURON bash in the folder "
               "L5bPCmodelsEH/mod and rerun example script")
        if not mod_pth in neuron.nrn_dll_loaded:
            neuron.h.nrn_load_dll(join(mod_pth, "nrnmech.dll"))
        neuron.nrn_dll_loaded.append(mod_pth)
    else:
        os.system('''
                  cd {}
                  nrnivmodl
                  '''.format(mod_pth))
        neuron.load_mechanisms(mod_pth)


def download_hallermann_model():

    print("Downloading Hallermann model")
    if sys.version < '3':
        from urllib2 import urlopen
    else:
        from urllib.request import urlopen
    import ssl
    from warnings import warn
    import zipfile
    #get the model files:
    u = urlopen('https://modeldb.science/download/144526',
                context=ssl._create_unverified_context())
    localFile = open(join(cell_models_folder, 'HallermannEtAl2012_r1.zip'), 'wb')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile(join(cell_models_folder, 'HallermannEtAl2012_r1.zip'), 'r')
    myzip.extractall(cell_models_folder)
    myzip.close()

    # Remove NEURON GUI from model files:
    model_file_ = open(join(hallermann_folder, "Cell parameters.hoc"), 'r')
    new_lines = ""
    for line in model_file_:
        changes = line.replace('load_proc("nrn', '//load_proc("nrn')
        changes = changes.replace('load_file("nrn', '//load_file("nrn')
        new_lines += changes
    new_lines += "parameters()\ngeom_nseg()\ninit_channels()\n"

    model_file_.close()
    model_file_mod = open(join(hallermann_folder, "Cell parameters_mod.hoc"), 'w')
    model_file_mod.write(new_lines)
    model_file_mod.close()

    #compile mod files every time, because of incompatibility with Mainen96 files:
    mod_pth = join(hallermann_folder)

    if "win32" in sys.platform:
        warn("no autompile of NMODL (.mod) files on Windows.\n"
             + "Run mknrndll from NEURON bash in the folder "
               "L5bPCmodelsEH/mod and rerun example script")
        if not mod_pth in neuron.nrn_dll_loaded:
            neuron.h.nrn_load_dll(join(mod_pth, "nrnmech.dll"))
        neuron.nrn_dll_loaded.append(mod_pth)
    else:
        os.system('''
                  cd {}
                  nrnivmodl
                  '''.format(mod_pth))
        neuron.load_mechanisms(mod_pth)


def download_BBP_model(cell_name="L5_TTPC2_cADpyr232_1"):

    os.makedirs(bbp_folder, exist_ok=True)
    os.makedirs(bbp_mod_folder, exist_ok=True)

    print("Downloading BBP model: ", cell_name)
    url = "https://bbp.epfl.ch/nmc-portal/assets/documents/static/downloads-zip/{}.zip".format(cell_name)

    if sys.version < '3':
        from urllib2 import urlopen
    else:
        from urllib.request import urlopen
    import ssl
    import zipfile
    #get the model files:
    u = urlopen(url, context=ssl._create_unverified_context())

    localFile = open(join(bbp_folder, '{}.zip'.format(cell_name)), 'wb')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile(join(bbp_folder, '{}.zip'.format(cell_name)), 'r')
    myzip.extractall(bbp_folder)
    myzip.close()
    os.remove(join(bbp_folder, '{}.zip'.format(cell_name)))
    #compile mod files every time, because of incompatibility with Mainen96 files:
    # mod_pth = join(hay_folder, "mod/")
    #
    # if "win32" in sys.platform:
    #     warn("no autompile of NMODL (.mod) files on Windows.\n"
    #          + "Run mknrndll from NEURON bash in the folder "
    #            "L5bPCmodelsEH/mod and rerun example script")
    #     if not mod_pth in neuron.nrn_dll_loaded:
    #         neuron.h.nrn_load_dll(join(mod_pth, "nrnmech.dll"))
    #     neuron.nrn_dll_loaded.append(mod_pth)
    # else:
    #     os.system('''
    #               cd {}
    #               nrnivmodl
    #               '''.format(mod_pth))
    #     neuron.load_mechanisms(mod_pth)

    # attempt to set up a folder with all unique mechanism mod files, compile, and
    # load them all
    compile_bbp_mechanisms(cell_name)


def compile_bbp_mechanisms(cell_name):
    from warnings import warn

    if not os.path.isdir(bbp_mod_folder):
        os.mkdir(bbp_mod_folder)
    cell_folder = join(bbp_folder, cell_name)
    for nmodl in glob(join(cell_folder, 'mechanisms', '*.mod')):
        while not os.path.isfile(join(bbp_mod_folder, os.path.split(nmodl)[-1])):
            if "win32" in sys.platform:
                os.system("copy {} {}".format(nmodl, bbp_mod_folder))
            else:
                os.system('cp {} {}'.format(nmodl,
                                            join(bbp_mod_folder, '.')))
    CWD = os.getcwd()
    os.chdir(bbp_mod_folder)
    if "win32" in sys.platform:
        warn("no autompile of NMODL (.mod) files on Windows. " +
             "Run mknrndll from NEURON bash in the folder %s" % bbp_mod_folder +
             "and rerun example script")
    else:
        os.system('nrnivmodl')
    os.chdir(CWD)


def posixpth(pth):
    """
    Replace Windows path separators with posix style separators
    """
    return pth.replace(os.sep, posixpath.sep)


def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within

    Arguments
    ---------
    f : file, mode 'r'

    Returns
    -------
    templatename : str

    '''
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            continue
    return templatename


def return_BBP_neuron(cell_name, tstop, dt):

    # load some required neuron-interface files
    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    CWD = os.getcwd()
    cell_folder = join(join(bbp_folder, cell_name))
    if not os.path.isdir(cell_folder):
        download_BBP_model(cell_name)

    neuron.load_mechanisms(bbp_mod_folder)
    os.chdir(cell_folder)
    add_synapses = False
    # get the template name
    f = open("template.hoc", 'r')
    templatename = get_templatename(f)
    f.close()

    # get biophys template name
    f = open("biophysics.hoc", 'r')
    biophysics = get_templatename(f)
    f.close()

    # get morphology template name
    f = open("morphology.hoc", 'r')
    morphology = get_templatename(f)
    f.close()

    # get synapses template name
    f = open(posixpth(os.path.join("synapses", "synapses.hoc")), 'r')
    synapses = get_templatename(f)
    f.close()

    neuron.h.load_file('constants.hoc')

    if not hasattr(neuron.h, morphology):
        """Create the cell model"""
        # Load morphology
        neuron.h.load_file(1, "morphology.hoc")
    if not hasattr(neuron.h, biophysics):
        # Load biophysics
        neuron.h.load_file(1, "biophysics.hoc")
    if not hasattr(neuron.h, synapses):
        # load synapses
        neuron.h.load_file(1, posixpth(os.path.join('synapses', 'synapses.hoc')
                                       ))
    if not hasattr(neuron.h, templatename):
        # Load main cell template
        neuron.h.load_file(1, "template.hoc")

    templatefile = posixpth(os.path.join(cell_folder, 'template.hoc'))

    morphologyfile = glob(os.path.join('morphology', '*'))[0]


    # Instantiate the cell(s) using LFPy
    cell = LFPy.TemplateCell(morphology=morphologyfile,
                             templatefile=templatefile,
                             templatename=templatename,
                             templateargs=1 if add_synapses else 0,
                             tstop=tstop,
                             dt=dt,
                             nsegs_method=None)
    os.chdir(CWD)
    # set view as in most other examples
    cell.set_rotation(x=np.pi / 2)
    return cell


def return_hay_cell(tstop, dt, make_passive=False):
    if not os.path.isfile(join(hay_folder, 'morphologies', 'cell1.asc')):
        download_hay_model()

    if make_passive:
        cell_params = {
            'morphology': join(hay_folder, 'morphologies', 'cell1.asc'),
            'passive': True,
            'passive_parameters': {"g_pas": 1 / 30000,
                                   "e_pas": -70.},
            'nsegs_method': "lambda_f",
            "Ra": 150,
            "cm": 1.0,
            "lambda_f": 100,
            'dt': dt,
            'tstart': -1,
            'tstop': tstop,
            'v_init': -70,
            'pt3d': True,
        }

        cell = LFPy.Cell(**cell_params)
        cell.set_rotation(x=4.729, y=-3.166)

        return cell
    else:
        if not hasattr(neuron.h, "CaDynamics_E2"):
            neuron.load_mechanisms(join(hay_folder, 'mod'))
        cell_params = {
            'morphology': join(hay_folder, "morphologies", "cell1.asc"),
            'templatefile': [join(hay_folder, 'models', 'L5PCbiophys3.hoc'),
                             join(hay_folder, 'models', 'L5PCtemplate.hoc')],
            'templatename': 'L5PCtemplate',
            'templateargs': join(hay_folder, 'morphologies', 'cell1.asc'),
            'passive': False,
            'nsegs_method': None,
            'dt': dt,
            'tstart': -200,
            'tstop': tstop,
            'v_init': -75,
            'celsius': 34,
            'pt3d': True,
        }

        cell = LFPy.TemplateCell(**cell_params)

        cell.set_rotation(x=4.729, y=-3.166)
        return cell


def return_hallermann_cell(tstop, dt, ):

    #model_folder = join(cell_models_folder, 'HallermannEtAl2012')
    if not os.path.isfile(join(hallermann_folder, '28_04_10_num19.hoc')):
        download_hallermann_model()
    neuron.load_mechanisms(hallermann_folder)

    # Define cell parameters
    cell_parameters = {  # various cell parameters,
        'morphology': join(hallermann_folder, '28_04_10_num19.hoc'),
        'v_init': -80.,  # initial crossmembrane potential
        'passive': False,  # switch on passive mechs
        'nsegs_method': 'lambda_f',
        'lambda_f': 500.,
        'dt': dt,  # [ms] dt's should be in powers of 2 for both,
        'tstart': -100,  # start time of simulation, recorders start at t=0
        'tstop': tstop,
        "extracellular": False,
        "pt3d": True,
        'custom_code': [join(hallermann_folder, 'Cell parameters_mod.hoc'),
                        #join(hallermann_folder, 'charge.hoc')
                        ]
    }

    cell = LFPy.Cell(**cell_parameters)
    cell.set_rotation(x=np.pi / 2, y=-0.1, z=0)
    return cell


def return_stick_cell(tstop, dt):

    h("forall delete_section()")
    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create axon[1]

    proc topol() { local i
      basic_shape()
    }
    proc basic_shape() {
      axon[0] {pt3dclear()
      pt3dadd(0, 0, 0, 1)
      pt3dadd(0, 0, 1000, 1)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        axon[0] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    forall {nseg = 200}
    }
    proc biophys() {
    }
    celldef()

    Ra = 150.
    cm = 1.
    Rm = 30000.

    forall {
        insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
        g_pas = 1 / Rm
        }
    """)
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,
                'tstart': 0.,
                'tstop': tstop,
            }
    cell = LFPy.Cell(**cell_params)
    cell.set_pos(x=-cell.x[0, 0])
    return cell


def return_ball_and_stick_cell(tstop, dt, apic_diam=2):

    h("forall delete_section()")
    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create soma[1], dend[1]

    proc topol() { local i
      basic_shape()
      connect dend(0), soma(1)
    }
    proc basic_shape() {
      soma[0] {pt3dclear()
      pt3dadd(0, 0, -10., 20.)
      pt3dadd(0, 0, 10., 20.)}
      dend[0] {pt3dclear()
      pt3dadd(0, 0, 10., %s)
      pt3dadd(0, 0, 1000, %s)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        soma[0] all.append()
        dend[0] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    soma[0] {nseg = 1}
    dend[0] {nseg = 200}
    }
    proc biophys() {
    }
    celldef()

    Ra = 150.
    cm = 1.
    Rm = 30000.

    forall {
        insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
        g_pas = 1 / Rm
        }
    """ % (apic_diam, apic_diam))
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,
                'tstart': -100.,
                'tstop': tstop,
                'pt3d': True,
            }
    cell = LFPy.Cell(**cell_params)
    # cell.set_pos(x=-cell.xstart[0])
    return cell


def return_two_comp_cell(tstop, dt):

    h("forall delete_section()")
    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create soma[1], dend[1]

    proc topol() { local i
      basic_shape()
      connect dend(0), soma(1)
    }
    proc basic_shape() {
      soma[0] {pt3dclear()
      pt3dadd(0, 0, -10.0, 20)
      pt3dadd(0, 0, 10., 20)}
      dend[0] {pt3dclear()
      pt3dadd(0, 0, 10.0, 20)
      pt3dadd(0, 0, 30.0, 20)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        soma[0] all.append()
        dend[0] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    soma[0] {nseg = 1}
    dend[0] {nseg = 1}
    }
    proc biophys() {
    }
    celldef()

    Ra = 150.
    cm = 1.
    Rm = 30000.

    forall {
        insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
        g_pas = 1 / Rm
        Ra = Ra
        cm = cm
        }
    """)
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,
                'tstart': -100.,
                'tstop': tstop,
                'pt3d': True,
            }
    cell = LFPy.Cell(**cell_params)
    # cell.set_pos(x=-cell.xstart[0])
    return cell


def remove_active_mechanisms(remove_list, cell):
    # remove_list = ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
    # "SK_E2", "K_Tst", "K_Pst",
    # "Im", "Ih", "CaDynamics_E2", "Ca_LVAst", "Ca", "Ca_HVA"]
    mt = h.MechanismType(0)
    for sec in h.allsec():
        for seg in sec:
            for mech in remove_list:
                mt.select(mech)
                mt.remove(sec=sec)
    return cell


def return_freq_and_psd(tvec, sig):
    """ Returns the power and freqency of the input signal"""
    import scipy.fftpack as ff
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    timestep = (tvec[1] - tvec[0])/1000. if type(tvec) in [list, np.ndarray] else tvec
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]

    Y = ff.fft(sig, axis=1)[:, pidxs[0]]

    power = np.abs(Y)**2/Y.shape[1]
    return freqs, power


def return_freq_and_amplitude(tvec, sig):
    """ Returns the amplitude and frequency of the input signal"""
    import scipy.fftpack as ff
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    timestep = (tvec[1] - tvec[0])/1000. if type(tvec) in [list, np.ndarray] else tvec
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]

    Y = ff.fft(sig, axis=1)[:, pidxs[0]]

    amplitude = np.abs(Y)/Y.shape[1]
    return freqs, amplitude


def return_freq_and_psd_welch(sig, welch_dict):
    from matplotlib import mlab as ml
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    psd = []
    freqs = None
    for idx in range(sig.shape[0]):
        yvec_w, freqs = ml.psd(sig[idx, :], **welch_dict)
        psd.append(yvec_w)
    return freqs, np.array(psd)


def make_WN_input(cell, max_freq):
    """ White Noise input ala Linden 2010 is made """
    tot_ntsteps = round((cell.tstop - cell.tstart) / cell.dt + 1)
    I = np.zeros(tot_ntsteps)
    tvec = np.arange(tot_ntsteps) * cell.dt
    for freq in range(1, max_freq + 1):
        I += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
    return I


def make_white_noise_stimuli(cell, input_idx, weight=None, max_freq=1100):

    input_scaling = 0.005
    np.random.seed(1234)
    input_array = input_scaling * (make_WN_input(cell, max_freq))

    noise_vec = (neuron.h.Vector(input_array) if weight is None
                 else neuron.h.Vector(input_array * weight))

    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print("Input inserted in ", sec.name())
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if syn is None:
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noise_vec.play(syn._ref_amp, cell.dt)
    return cell, syn, noise_vec


def point_axon_down(cell):
    '''
    Make the axon of Hay model point downwards, start at soma mid point,
    comp for soma diameter
    
    Keyword arguments:
    :
        cell : LFPy.TemplateCell instance
    
    '''
    iaxon = cell.get_idx(section='axon')
    isoma = cell.get_idx(section='soma')
    cell.x[iaxon, 0] = cell.x[isoma].mean(axis=1)
    cell.x[iaxon, 1] = cell.x[isoma].mean(axis=1)
    
    cell.y[iaxon, 0] = cell.y[isoma].mean(axis=1)
    cell.y[iaxon, 1] = cell.y[isoma].mean(axis=1)
    
    j = 0
    for i in iaxon:
        cell.z[i, 0] = cell.z[isoma].mean(axis=1) \
                - cell.d[isoma]/2 - cell.length[i] * j
        cell.z[i, 1] = cell.z[isoma].mean(axis=1) \
                - cell.d[isoma]/2 - cell.length[i] - cell.length[i]*j
        j += 1
    
    ##point the pt3d axon as well
    for sec in cell.allseclist:
        if sec.name().rfind('axon') >= 0:
            x0 = cell.x[cell.get_idx(sec.name())[0], 0]
            y0 = cell.y[cell.get_idx(sec.name())[0], 0]
            z0 = cell.z[cell.get_idx(sec.name())[0], 0]
            L = sec.L
            for j in range(int(neuron.h.n3d(sec=sec))):
                neuron.h.pt3dchange(j, x0, y0, z0,
                                 sec.diam, sec=sec)
                z0 -= L / (neuron.h.n3d(sec=sec)-1)

    # let NEURON know about the changes we just did:
    neuron.h.define_shape()
    cell.x3d, cell.y3d, cell.z3d, cell.diam3d, cell.ar3d = cell._collect_pt3d()


def return_equidistal_xyz(num_points, r):
    """Algorithm to calculate num_points equidistial points on the surface of
    a sphere. Note that the returned number of points might slightly deviate
    from expected number of points.

    Algorith from: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    """
    a = 4 * np.pi / num_points

    d = np.sqrt(a)
    M_theta = int(np.round(np.pi / d))

    d_theta = np.pi / M_theta
    d_phi = a / d_theta

    xs = []
    ys = []
    zs = []

    i = 0
    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta
        M_phi = int(np.round(2 * np.pi * np.sin(theta) / d_phi))
        for n in range(M_phi):
            phi = 2 * np.pi * n / M_phi
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            i += 1
            # if i == num_points:
            #     return xs, ys, zs
    return np.array(xs), np.array(ys), np.array(zs)


def align_cell_to_axes(cell):
    """
    Based on code from: https://github.com/lastis/LFPy_util/
    Rotates the cell such that **y_axis** is paralell to the global y-axis and
    **x_axis** will be aligned to the global x-axis as well as possible.
    **y_axis** and **x_axis** should be orthogonal, but need not be.
    :param `~LFPy.Cell` cell:
        Initialized Cell object to rotate.

    """

    from sklearn.decomposition import PCA
    points = np.array([cell.x.mean(axis=1),
                       cell.y.mean(axis=1),
                       cell.z.mean(axis=1)])
    pca = PCA(n_components=3)
    pca.fit(points[:3].T)
    axes = pca.components_

    y_axis = np.asarray(axes[2])
    y_axis = y_axis / np.linalg.norm(y_axis)

    dx = y_axis[0]
    dy = y_axis[1]
    dz = y_axis[2]

    x_angle = -np.arctan2(dz, dy)
    z_angle = np.arctan2(dx, np.sqrt(dy * dy + dz * dz))

    cell.set_rotation(x_angle, None, z_angle)
    x_axis = axes[1]
    if x_axis is None:
        return

    x_axis = np.asarray(x_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    Rx = rotation_matrix([1, 0, 0], x_angle)
    Rz = rotation_matrix([0, 0, 1], z_angle)

    x_axis = np.dot(x_axis, Rx)
    x_axis = np.dot(x_axis, Rz)

    dx = x_axis[0]
    dz = x_axis[2]

    y_angle = np.arctan2(dz, dx)
    cell.set_rotation(None, y_angle, None)

    if np.abs(np.min(cell.z)) > np.abs(np.max(cell.z)):
        cell.set_rotation(x=np.pi)


def rotation_matrix(axis, theta):
    """
    Based on code from: https://github.com/lastis/LFPy_util/
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    Uses the Euler-rodrigues formula
    """
    theta = -theta
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
