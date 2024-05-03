import xraydb
import pandas as pd
import numpy as np

qas_foils = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
             'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Sn',
             'W', 'Re', 'Ir', 'Pt', 'Au', 'Pb']

_qas_foil_data = []
for element in qas_foils:
    for edge in xraydb.xray_edges(element):
        energy = xraydb.xray_edges(element)[edge].energy
        if (energy >= 4500) and (energy <= 32000):
            _qas_foil_data.append({'element': element,
                          'edge':edge,
                          'energy':energy})

qas_foils_df = pd.DataFrame(_qas_foil_data)

def find_correct_foil(energy= None,  element='Cu', edge='K'):
    if not energy:
        energy = xraydb.xray_edge(element, edge).energy

    # Finding foils options for corresponding element and edge
    foils_options = qas_foils_df.loc[(qas_foils_df['energy'] > energy - 200) &
                                     (qas_foils_df['energy'] < energy + 600)]

    df2 = foils_options.to_string(index=False)

    # if no element is found going to empty holder
    if len(foils_options) == 0:
        foil = None
        foil_edge = None
        foil_energy = None

    # # Among the foils_options first select the foil with corresponding element and edge
    indx = foils_options.index[(foils_options['element'] == element) & (foils_options['edge'] == edge)]

    # second if above condition doesnot match than search for foil whose K edge is near to the element of interest
    indx_k = foils_options.index[(foils_options['edge'] == "K")]

    # third if above condition doesnot match than search for foil whose L3 edge is near to the element of interest
    indx_l3 = foils_options.index[(foils_options['edge'] == "L3")]

    # fourth if above condition doesnot match than search for foil whose L2 edge is near to the element of interest
    indx_l2 = foils_options.index[(foils_options['edge'] == "L2")]

    # fifth if above condition doesnot match than search for foil whose L1 edge is near to the element of interest
    indx_l1 = foils_options.index[(foils_options['edge'] == "L1")]

    if len(indx) == 1:
        foil = str(foils_options['element'][indx[0]])
        foil_edge = str(foils_options['edge'][indx[0]])
        foil_energy = str(foils_options['energy'][indx[0]])
    elif len(indx_k) >= 1:
        foil = str(foils_options['element'][indx_k[0]])
        foil_edge = str(foils_options['edge'][indx_k[0]])
        foil_energy = str(foils_options['energy'][indx_k[0]])
    elif len(indx_l3) >= 1:
        foil = str(foils_options['element'][indx_l3[0]])
        foil_edge = str(foils_options['edge'][indx_l3[0]])
        foil_energy = str(foils_options['energy'][indx_l3[0]])
    elif len(indx_l2) >= 1:
        foil = str(foils_options['element'][indx_l2[0]])
        foil_edge = str(foils_options['edge'][indx_l2[0]])
        foil_energy = str(foils_options['energy'][indx_l2[0]])
    elif len(indx_l1) >= 1:
        foil = foils_options['element'][indx_l1[0]]
        foil_edge = str(foils_options['edge'][indx_l1[0]])
        foil_energy = str(foils_options['energy'][indx_l1[0]])

    if foil is not None:
        foil_energy = float(foil_energy)

    return foil,foil_edge,foil_energy
def get_atomic_symbol() -> list:
    atomic_number: list = np.arange(22,93).tolist()
    return [xraydb.atomic_symbol(n) for n in atomic_number]

def get_possible_edges(element: str = 'Cu') -> dict:
    possible_edges: dict = {}
    edges: dict = xraydb.xray_edges(element)
    for edge, value in edges.items():
        if value.energy < 32000 and value.energy > 4500:
            possible_edges[edge] = value.energy
    return possible_edges

def get_ionchamber_absorption(gases:dict[str], energy:float):
    flux = xraydb.ionchamber_fluxes(gas=gases, volts=1, length=14.5, energy=energy, sensitivity=1e-6)
    absorption = (100 * (flux.incident - flux.transmitted)/flux.incident)
    return absorption


def create_gases_dict(flow, absorption, energy, gas1='nitrogen', gas2='helium'):
    gases_dict = {'gases_initial': {gas1:5*flow, gas2:5*(100 - flow)},
                  'gases_final'  : {gas1:1*flow, gas2:1*(100 - flow)},
                  'parameters'   : {'absorption': np.round(absorption, 2), 'energy': energy}
                  }
    return gases_dict

def get_ionchamber_gases_i0(element: str, edge:str, channel='i0'):
    flow_range = {'nitrogen' : np.arange(10, 101, 1).tolist()}
    energy = xraydb.xray_edge(element=element, edge=edge).energy
    gases = None

    for n2_flow in flow_range['nitrogen']:
        absorption_n2_he = get_ionchamber_absorption(gases={'nitrogen':n2_flow, 'helium':100 - n2_flow},
                                                     energy=energy)
        absorption_n2_ar = get_ionchamber_absorption(gases={'nitrogen': n2_flow, 'argon': 100 - n2_flow},
                                                     energy=energy)

        if abs(10-absorption_n2_he) < 1:
            gases = create_gases_dict(flow=n2_flow, absorption=absorption_n2_he, gas1='nitrogen', gas2='helium', energy=energy)
            break
        elif abs(10-absorption_n2_ar) < 1:
            gases = create_gases_dict(flow=n2_flow, absorption=absorption_n2_ar, gas1='nitrogen', gas2='argon', energy=energy)
            break
        else:
            gases = None

    if gases is None:
        _gas = {'nirogen':10, 'argon':90}
        _abs = get_ionchamber_absorption(gases=_gas, energy=energy)
        gases = create_gases_dict(flow=10, absorption=_abs, gas1='nitrogen', gas2='argon', energy=energy)

    return gases


def get_ionchamber_gases_it(element: str, edge:str, channel='it'):
    flow_range = {'nitrogen' : np.arange(10, 101, 1).tolist()}
    energy = xraydb.xray_edge(element=element, edge=edge).energy
    gases = None

    for n2_flow in flow_range['nitrogen']:
        absorption_n2_ar = get_ionchamber_absorption(gases={'nitrogen': n2_flow, 'argon': 100 - n2_flow},
                                                     energy=energy)
        if abs(30-absorption_n2_ar) < 2:
            gases = create_gases_dict(flow=n2_flow, absorption=absorption_n2_ar, gas1='nitrogen', gas2='argon', energy=energy)
            break
        else:
            gases = None

    if gases is None:
        if energy < 6000:
            _gas = {'nitrogen':100}
            _abs = get_ionchamber_absorption(gases=_gas, energy=energy)
            gases = create_gases_dict(flow=100, absorption=_abs, gas1='nitrogen', gas2='argon', energy=energy)
        else:
            _gas = {'nitrogen': 10, 'argon' :90}
            _abs = get_ionchamber_absorption(gases=_gas, energy=energy)
            gases = create_gases_dict(flow=10, absorption=_abs, gas1='nitrogen', gas2='argon', energy=energy)

    return gases

atomic_dict = {}
for atomic_symbol in get_atomic_symbol():
    atomic_dict[atomic_symbol] = {}
    for edge in get_possible_edges(atomic_symbol):
        atomic_dict[atomic_symbol][edge] = {}
        atomic_dict[atomic_symbol][edge]['i0'] = get_ionchamber_gases_i0(element=atomic_symbol, edge=edge)
        atomic_dict[atomic_symbol][edge]['it'] = get_ionchamber_gases_it(element=atomic_symbol, edge=edge)


{'Ti': {'K': {'nitrogen': 16, 'helium': 84, 'absorption': 9.004357511914268}},
 'V': {'K': {'nitrogen': 22, 'helium': 78, 'absorption': 9.22098112182543}},
 'Cr': {'K': {'nitrogen': 29, 'helium': 71, 'absorption': 9.186335728182225}},
 'Mn': {'K': {'nitrogen': 38, 'helium': 62, 'absorption': 9.199418813429869}},
 'Fe': {'K': {'nitrogen': 48, 'helium': 52, 'absorption': 9.001806527897454}},
 'Co': {'K': {'nitrogen': 62, 'helium': 38, 'absorption': 9.08987004067498}},
 'Ni': {'K': {'nitrogen': 78, 'helium': 22, 'absorption': 9.032311396474832}},
 'Cu': {'K': {'nitrogen': 98, 'helium': 2, 'absorption': 9.055575146606683}},
 'Zn': {'K': {'nitrogen': 98, 'argon': 2, 'absorption': 10.630404539567879}},
 'Ga': {'K': {'nitrogen': 97, 'argon': 3, 'absorption': 10.020940044779795}},
 'Ge': {'K': {'nitrogen': 95, 'argon': 5, 'absorption': 10.40681949163222}},
 'As': {'K': {'nitrogen': 93, 'argon': 7, 'absorption': 10.423047333713956}},
 'Se': {'K': {'nitrogen': 90, 'argon': 10, 'absorption': 10.915516339661508}},
 'Br': {'K': {'nitrogen': 88, 'argon': 12, 'absorption': 10.42993821961353}},
 'Kr': {'K': {'nitrogen': 84, 'argon': 16, 'absorption': 10.861136090988785}},
 'Rb': {'K': {'nitrogen': 80, 'argon': 20, 'absorption': 10.958957454769523}},
 'Sr': {'K': {'nitrogen': 76, 'argon': 24, 'absorption': 10.815391956106696}},
 'Y': {'K': {'nitrogen': 71, 'argon': 29, 'absorption': 10.817205670264222}},
 'Zr': {'K': {'nitrogen': 65, 'argon': 35, 'absorption': 10.888636438868039}},
 'Nb': {'K': {'nitrogen': 58, 'argon': 42, 'absorption': 10.978778606540146}},
 'Mo': {'K': {'nitrogen': 51, 'argon': 49, 'absorption': 10.872041149985392}},
 'Tc': {'K': {'nitrogen': 42, 'argon': 58, 'absorption': 10.949615051713764}},
 'Ru': {'K': {'nitrogen': 32, 'argon': 68, 'absorption': 10.985603664280575}},
 'Rh': {'K': {'nitrogen': 21, 'argon': 79, 'absorption': 10.97695695608987}},
 'Pd': {'K': {'nitrogen': 10, 'argon': 90, 'absorption': 10.823112583109042}},
 'Ag': {'K': {'nitrogen': 10, 'argon': 90, 'absorption': 9.535066836575421}},
 'Cd': {'K': {'nitrogen': 10, 'argon': 90, 'absorption': 8.416867363179522}},
 'In': {'K': {'nitrogen': 10, 'argon': 90, 'absorption': 7.446502941413719}},
 'Sn': {'K': {'nitrogen': 10, 'argon': 90, 'absorption': 6.604354375803428}},
 'Sb': {'K': {'nitrogen': 10, 'argon': 90, 'absorption': 5.87249422107421},
  'L1': {'nitrogen': 14, 'helium': 86, 'absorption': 9.316060914916157}},
 'Te': {'K': {'nitrogen': 10, 'argon': 90, 'absorption': 5.235074060541383},
  'L1': {'nitrogen': 16, 'helium': 84, 'absorption': 9.146444737887261},
  'L2': {'nitrogen': 13, 'helium': 87, 'absorption': 9.164460252601197}},
 'I': {'L1': {'nitrogen': 19, 'helium': 81, 'absorption': 9.330887333913635},
  'L2': {'nitrogen': 15, 'helium': 85, 'absorption': 9.06003946399137},
  'L3': {'nitrogen': 13, 'helium': 87, 'absorption': 9.48451681525262}},
 'Xe': {'L1': {'nitrogen': 22, 'helium': 78, 'absorption': 9.279476119772019},
  'L2': {'nitrogen': 18, 'helium': 82, 'absorption': 9.279380019490352},
  'L3': {'nitrogen': 15, 'helium': 85, 'absorption': 9.422881574988699}},
 'Cs': {'L1': {'nitrogen': 25, 'helium': 75, 'absorption': 9.147086002563098},
  'L2': {'nitrogen': 21, 'helium': 79, 'absorption': 9.338100378517437},
  'L3': {'nitrogen': 17, 'helium': 83, 'absorption': 9.282152909146939}},
 'Ba': {'L1': {'nitrogen': 29, 'helium': 71, 'absorption': 9.186335728182225},
  'L2': {'nitrogen': 24, 'helium': 76, 'absorption': 9.214247249888247},
  'L3': {'nitrogen': 19, 'helium': 81, 'absorption': 9.032455354853772}},
 'La': {'L1': {'nitrogen': 33, 'helium': 67, 'absorption': 9.1075477958312},
  'L2': {'nitrogen': 27, 'helium': 73, 'absorption': 9.006722513347821},
  'L3': {'nitrogen': 22, 'helium': 78, 'absorption': 9.134130200274486}},
 'Ce': {'L1': {'nitrogen': 38, 'helium': 62, 'absorption': 9.163003465388933},
  'L2': {'nitrogen': 31, 'helium': 69, 'absorption': 9.00250616996689},
  'L3': {'nitrogen': 25, 'helium': 75, 'absorption': 9.105702048237667}},
 'Pr': {'L1': {'nitrogen': 43, 'helium': 57, 'absorption': 9.098441275060829},
  'L2': {'nitrogen': 36, 'helium': 64, 'absorption': 9.135301560969015},
  'L3': {'nitrogen': 29, 'helium': 71, 'absorption': 9.297733314692197}},
 'Nd': {'L1': {'nitrogen': 49, 'helium': 51, 'absorption': 9.126969145898084},
  'L2': {'nitrogen': 41, 'helium': 59, 'absorption': 9.126914364785883},
  'L3': {'nitrogen': 32, 'helium': 68, 'absorption': 9.087654791327878}},
 'Pm': {'L1': {'nitrogen': 55, 'helium': 45, 'absorption': 9.032108622199283},
  'L2': {'nitrogen': 46, 'helium': 54, 'absorption': 9.00376301996465},
  'L3': {'nitrogen': 36, 'helium': 64, 'absorption': 9.057989067215305}},
 'Sm': {'L1': {'nitrogen': 63, 'helium': 37, 'absorption': 9.132685878085463},
  'L2': {'nitrogen': 53, 'helium': 47, 'absorption': 9.125909358412443},
  'L3': {'nitrogen': 41, 'helium': 59, 'absorption': 9.150449340194871}},
 'Eu': {'L1': {'nitrogen': 71, 'helium': 29, 'absorption': 9.11795363718295},
  'L2': {'nitrogen': 60, 'helium': 40, 'absorption': 9.122092484457237},
  'L3': {'nitrogen': 46, 'helium': 54, 'absorption': 9.13837588757007}},
 'Gd': {'L1': {'nitrogen': 79, 'helium': 21, 'absorption': 9.008012964875025},
  'L2': {'nitrogen': 67, 'helium': 33, 'absorption': 9.018247390909458},
  'L3': {'nitrogen': 51, 'helium': 49, 'absorption': 9.043912974433702}},
 'Tb': {'L1': {'nitrogen': 89, 'helium': 11, 'absorption': 9.022126685368008},
  'L2': {'nitrogen': 76, 'helium': 24, 'absorption': 9.063281473675874},
  'L3': {'nitrogen': 57, 'helium': 43, 'absorption': 9.037654338691594}},
 'Dy': {'L1': {'nitrogen': 99, 'argon': 1, 'absorption': 10.901782707758878},
  'L2': {'nitrogen': 85, 'helium': 15, 'absorption': 9.008466385808372},
  'L3': {'nitrogen': 64, 'helium': 36, 'absorption': 9.089518194269628}},
 'Ho': {'L1': {'nitrogen': 99, 'argon': 1, 'absorption': 9.811471578622523},
  'L2': {'nitrogen': 96, 'helium': 4, 'absorption': 9.054876417433434},
  'L3': {'nitrogen': 71, 'helium': 29, 'absorption': 9.056489827176872}},
 'Er': {'L1': {'nitrogen': 98, 'argon': 2, 'absorption': 10.35595351249781},
  'L2': {'nitrogen': 99, 'argon': 1, 'absorption': 10.201077258118803},
  'L3': {'nitrogen': 79, 'helium': 21, 'absorption': 9.063697330556195}},
 'Tm': {'L1': {'nitrogen': 97, 'argon': 3, 'absorption': 10.7168879320956},
  'L2': {'nitrogen': 98, 'argon': 2, 'absorption': 10.75890614131842},
  'L3': {'nitrogen': 87, 'helium': 13, 'absorption': 9.00637281832663}},
 'Yb': {'L1': {'nitrogen': 96, 'argon': 4, 'absorption': 10.938098115064323},
  'L2': {'nitrogen': 98, 'argon': 2, 'absorption': 9.71790432023304},
  'L3': {'nitrogen': 97, 'helium': 3, 'absorption': 9.068514872344283}},
 'Lu': {'L1': {'nitrogen': 96, 'argon': 4, 'absorption': 9.917144867518019},
  'L2': {'nitrogen': 97, 'argon': 3, 'absorption': 10.068812141357267},
  'L3': {'nitrogen': 99, 'argon': 1, 'absorption': 10.262830604487146}},
 'Hf': {'L1': {'nitrogen': 94, 'argon': 6, 'absorption': 10.98817192513441},
  'L2': {'nitrogen': 96, 'argon': 4, 'absorption': 10.250610865814492},
  'L3': {'nitrogen': 98, 'argon': 2, 'absorption': 10.933461196591825}},
 'Ta': {'L1': {'nitrogen': 93, 'argon': 7, 'absorption': 10.874516096608994},
  'L2': {'nitrogen': 95, 'argon': 5, 'absorption': 10.323313501547021},
  'L3': {'nitrogen': 98, 'argon': 2, 'absorption': 9.983960311072948}},
 'W': {'L1': {'nitrogen': 92, 'argon': 8, 'absorption': 10.70570653383506},
  'L2': {'nitrogen': 94, 'argon': 6, 'absorption': 10.299519232185995},
  'L3': {'nitrogen': 97, 'argon': 3, 'absorption': 10.457359267125533}},
 'Re': {'L1': {'nitrogen': 91, 'argon': 9, 'absorption': 10.489943341741249},
  'L2': {'nitrogen': 93, 'argon': 7, 'absorption': 10.207800273985239},
  'L3': {'nitrogen': 96, 'argon': 4, 'absorption': 10.800270353621462}},
 'Os': {'L1': {'nitrogen': 89, 'argon': 11, 'absorption': 10.89524515502951},
  'L2': {'nitrogen': 91, 'argon': 9, 'absorption': 10.816797343279903},
  'L3': {'nitrogen': 96, 'argon': 4, 'absorption': 9.914655373775586}},
 'Ir': {'L1': {'nitrogen': 88, 'argon': 12, 'absorption': 10.545249001239924},
  'L2': {'nitrogen': 90, 'argon': 10, 'absorption': 10.539796780756332},
  'L3': {'nitrogen': 95, 'argon': 5, 'absorption': 10.126999381280495}},
 'Pt': {'L1': {'nitrogen': 86, 'argon': 14, 'absorption': 10.73138776450713},
  'L2': {'nitrogen': 88, 'argon': 12, 'absorption': 10.859686734270014},
  'L3': {'nitrogen': 94, 'argon': 6, 'absorption': 10.251352523567629}},
 'Au': {'L1': {'nitrogen': 84, 'argon': 16, 'absorption': 10.806414317405038},
  'L2': {'nitrogen': 87, 'argon': 13, 'absorption': 10.475366398343137},
  'L3': {'nitrogen': 93, 'argon': 7, 'absorption': 10.300651068283956}},
 'Hg': {'L1': {'nitrogen': 82, 'argon': 18, 'absorption': 10.788926662320039},
  'L2': {'nitrogen': 85, 'argon': 15, 'absorption': 10.591027239865493},
  'L3': {'nitrogen': 92, 'argon': 8, 'absorption': 10.278822018793838}},
 'Tl': {'L1': {'nitrogen': 80, 'argon': 20, 'absorption': 10.679543086328612},
  'L2': {'nitrogen': 83, 'argon': 17, 'absorption': 10.604766375525806},
  'L3': {'nitrogen': 90, 'argon': 10, 'absorption': 10.915516339661508}},
 'Pb': {'L1': {'nitrogen': 77, 'argon': 23, 'absorption': 10.896473355227883},
  'L2': {'nitrogen': 80, 'argon': 20, 'absorption': 10.958957454769523},
  'L3': {'nitrogen': 89, 'argon': 11, 'absorption': 10.74549102484386}},
 'Bi': {'L1': {'nitrogen': 74, 'argon': 26, 'absorption': 10.998735055402747},
  'L2': {'nitrogen': 78, 'argon': 22, 'absorption': 10.795912744105989},
  'L3': {'nitrogen': 88, 'argon': 12, 'absorption': 10.545249001239924}},
 'Po': {'L1': {'nitrogen': 71, 'argon': 29, 'absorption': 10.987923939809017},
  'L2': {'nitrogen': 75, 'argon': 25, 'absorption': 10.91607788193423},
  'L3': {'nitrogen': 86, 'argon': 14, 'absorption': 10.869524097342998}},
 'At': {'L1': {'nitrogen': 68, 'argon': 32, 'absorption': 10.918896316038039},
  'L2': {'nitrogen': 72, 'argon': 28, 'absorption': 10.945996845505844},
  'L3': {'nitrogen': 85, 'argon': 15, 'absorption': 10.581025591966439}},
 'Rn': {'L1': {'nitrogen': 65, 'argon': 35, 'absorption': 10.806033468522482},
  'L2': {'nitrogen': 69, 'argon': 31, 'absorption': 10.89837325922574},
  'L3': {'nitrogen': 83, 'argon': 17, 'absorption': 10.759294605011675}},
 'Fr': {'L1': {'nitrogen': 61, 'argon': 39, 'absorption': 10.842817113970897},
  'L2': {'nitrogen': 66, 'argon': 34, 'absorption': 10.777003715897116},
  'L3': {'nitrogen': 81, 'argon': 19, 'absorption': 10.858630827743609},
  'M1': {'nitrogen': 14, 'helium': 86, 'absorption': 9.582079838658618}},
 'Ra': {'L1': {'nitrogen': 57, 'argon': 43, 'absorption': 10.809109214420658},
  'L2': {'nitrogen': 62, 'argon': 38, 'absorption': 10.851505609090923},
  'L3': {'nitrogen': 79, 'argon': 21, 'absorption': 10.902935266482812},
  'M1': {'nitrogen': 15, 'helium': 85, 'absorption': 9.222687690277464}},
 'Ac': {'L1': {'nitrogen': 52, 'argon': 48, 'absorption': 10.916928394672267},
  'L2': {'nitrogen': 58, 'argon': 42, 'absorption': 10.829063268256961},
  'L3': {'nitrogen': 77, 'argon': 23, 'absorption': 10.878056076156643},
  'M1': {'nitrogen': 17, 'helium': 83, 'absorption': 9.335499709407731},
  'M2': {'nitrogen': 14, 'helium': 86, 'absorption': 9.558561610685599}},
 'Th': {'L1': {'nitrogen': 47, 'argon': 53, 'absorption': 10.916479945283612},
  'L2': {'nitrogen': 53, 'argon': 47, 'absorption': 10.940627482313765},
  'L3': {'nitrogen': 75, 'argon': 25, 'absorption': 10.815665228419691},
  'M1': {'nitrogen': 19, 'helium': 81, 'absorption': 9.361946255393892},
  'M2': {'nitrogen': 15, 'helium': 85, 'absorption': 9.17894827333771}},
 'Pa': {'L1': {'nitrogen': 42, 'argon': 58, 'absorption': 10.86434607406154},
  'L2': {'nitrogen': 48, 'argon': 52, 'absorption': 10.96689133174553},
  'L3': {'nitrogen': 73, 'argon': 27, 'absorption': 10.71926856798311},
  'M1': {'nitrogen': 21, 'helium': 79, 'absorption': 9.298144222710263},
  'M2': {'nitrogen': 17, 'helium': 83, 'absorption': 9.340856150728946}},
 'U': {'L1': {'nitrogen': 36, 'argon': 64, 'absorption': 10.895293308992299},
  'L2': {'nitrogen': 43, 'argon': 57, 'absorption': 10.920909769182634},
  'L3': {'nitrogen': 70, 'argon': 30, 'absorption': 10.897569301434286},
  'M1': {'nitrogen': 23, 'helium': 77, 'absorption': 9.206347697467727},
  'M2': {'nitrogen': 19, 'helium': 81, 'absorption': 9.361946255393892}}}
