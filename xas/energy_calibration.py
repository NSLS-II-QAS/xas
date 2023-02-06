import xraydb
import pandas as pd

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