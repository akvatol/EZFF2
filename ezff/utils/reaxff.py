import numpy as np
import itertools

class reax_forcefield:

    def __init__(self,filename = None, template = 'ff.template.generated', ranges = 'param_ranges'):
        print('Opened new ReaxFF')
        self.params_write = []
        self.template = template
        self.ranges = ranges
        if not filename is None:
            self.read_forcefield(filename)

    def read_forcefield(self,filename):
        fffile = open(filename,'r')
        ff = fffile.readlines()
        fffile.close()
        self.full = ff
        return

    def split_forcefield(self):
        header, general, onebody, twobody, offdiagonal, threebody, fourbody, hbond = [], [], [], [], [], [], [], []
        counter = 0
        ff = self.full

        # Read HEADER line
        header = ff[0]
        counter += 1

        num_general = int(ff[counter].strip().split()[0])
        general_string = ff[counter:counter+num_general+1] # one for the header another for the number of parameters
        for line in general_string:
            general.append(line.strip().split())
        counter += (num_general + 1)

        num_onebody = int(ff[counter].strip().split()[0])
        onebody_string = ff[counter:counter+(num_onebody*4)+4] # one for the header another for the number of parameters
        for line in onebody_string:
            onebody.append(line.strip().split())
        counter += ((num_onebody*4) + 4)

        num_twobody = int(ff[counter].strip().split()[0])
        twobody_string = ff[counter:counter+(num_twobody*2)+2] # one for the header another for the number of parameters
        for line in twobody_string:
            twobody.append(line.strip().split())
        counter += ((num_twobody*2) + 2)

        num_offdiagonal = int(ff[counter].strip().split()[0])
        offdiagonal_string = ff[counter:counter+(num_offdiagonal*1)+1] # one for the header another for the number of parameters
        for line in offdiagonal_string:
            offdiagonal.append(line.strip().split())
        counter += ((num_offdiagonal*1) + 1)

        num_threebody = int(ff[counter].strip().split()[0])
        threebody_string = ff[counter:counter+(num_threebody*1)+1] # one for the header another for the number of parameters
        for line in threebody_string:
            threebody.append(line.strip().split())
        counter += ((num_threebody*1) + 1)

        num_fourbody = int(ff[counter].strip().split()[0])
        fourbody_string = ff[counter:counter+(num_fourbody*1)+1] # one for the header another for the number of parameters
        for line in fourbody_string:
            fourbody.append(line.strip().split())
        counter += ((num_fourbody*1) + 1)

        num_hbond = int(ff[counter].strip().split()[0])
        hbond_string = ff[counter:counter+(num_hbond*1)+1] # one for the header another for the number of parameters
        for line in hbond_string:
            hbond.append(line.strip().split())
        counter += ((num_hbond*1) + 1)

        self.header = header
        self.general = general
        self.onebody = onebody
        self.twobody = twobody
        self.offdiagonal = offdiagonal
        self.threebody = threebody
        self.fourbody = fourbody
        self.hbond = hbond

        return #header, general, onebody, twobody, offdiagonal, threebody, fourbody, hbond

    def get_element_number(self,element):
        new_onebody = self.onebody[4::4]
        for i in range(len(new_onebody)):
            if new_onebody[i][0].lower().upper() == element.lower().upper():
                return i+1
        return 0



    def template_bond_order(self, e1, e2, double_bond = False, triple_bond = False, bounds = 0.1):
        ie1, ie2 = self.get_element_number(e1), self.get_element_number(e2)

        #-------------------#
        #--- SINGLE BOND ---#
        #-------------------#

        # PBO1 and PBO2
        for index, line in enumerate(self.twobody[2::2]):
            if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                break
        line_number = (2 + (2*index) + 1)

        #PBO1
        PBO1 = float(self.twobody[line_number][13-8-1]) # 13th term, -8 for previous line, -1 for 0 indexing
        self.twobody[line_number][13-8-1] = '<<PBO1_'+e1+'_'+e2+'>>'
        self.params_write.append(['PBO1_'+e1+'_'+e2, str(PBO1*(1-bounds)), str(PBO1*(1+bounds))])

        #PBO2
        PBO2 = float(self.twobody[line_number][14-8-1]) # 14th term, -8 for previous line, -1 for 0 indexing
        self.twobody[line_number][14-8-1] = '<<PBO2_'+e1+'_'+e2+'>>'
        self.params_write.append(['PBO2_'+e1+'_'+e2, str(PBO2*(1-bounds)), str(PBO2*(1+bounds))])

        # ro_sigma
        for index, line in enumerate(self.offdiagonal[1:]):
            if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                break
        line_number = (1 + (1*index))

        # ro_sigma
        ro_sigma = float(self.offdiagonal[line_number][4+2-1])  # 4th term, +2 for atom indices, -1 for 0 indexing
        self.offdiagonal[line_number][4+2-1] = '<<ro_sigma_'+e1+'_'+e2+'>>'
        self.params_write.append(['ro_sigma_'+e1+'_'+e2, str(ro_sigma*(1-bounds)), str(ro_sigma*(1+bounds))])



        #-------------------#
        #--- DOUBLE BOND ---#
        #-------------------#
        if double_bond:
            # PBO1 and PBO2
            for index, line in enumerate(self.twobody[2::2]):
                if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                    break
            line_number = (2 + (2*index) + 1)

            #PBO3
            PBO3 = float(self.twobody[line_number][10-8-1]) # 10th term, -8 for previous line, -1 for 0 indexing
            self.twobody[line_number][10-8-1] = '<<PBO3_'+e1+'_'+e2+'>>'
            self.params_write.append(['PBO3_'+e1+'_'+e2, str(PBO3*(1-bounds)), str(PBO3*(1+bounds))])

            #PBO4
            PBO4 = float(self.twobody[line_number][11-8-1]) # 11th term, -8 for previous line, -1 for 0 indexing
            self.twobody[line_number][11-8-1] = '<<PBO4_'+e1+'_'+e2+'>>'
            self.params_write.append(['PBO4_'+e1+'_'+e2, str(PBO4*(1-bounds)), str(PBO4*(1+bounds))])

            # ro_pi
            for index, line in enumerate(self.offdiagonal[1:]):
                if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                    break
            line_number = (1 + (1*index))

            # ro_pi
            ro_pi = float(self.offdiagonal[line_number][5+2-1])  # 4th term, +2 for atom indices, -1 for 0 indexing
            self.offdiagonal[line_number][5+2-1] = '<<ro_pi_'+e1+'_'+e2+'>>'
            self.params_write.append(['ro_pi_'+e1+'_'+e2, str(ro_pi*(1-bounds)), str(ro_pi*(1+bounds))])


        #-------------------#
        #--- TRIPLE BOND ---#
        #-------------------#
        if triple_bond:
            # PBO5 and PBO6
            for index, line in enumerate(self.twobody[2::2]):
                if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                    break
            line_number = (2 + (2*index))

            #PBO5
            PBO5 = float(self.twobody[line_number][5+2-1]) # 5th term, +2 for atom indices, -1 for 0 indexing
            self.twobody[line_number][5+2-1] = '<<PBO5_'+e1+'_'+e2+'>>'
            self.params_write.append(['PBO5_'+e1+'_'+e2, str(PBO5*(1-bounds)), str(PBO5*(1+bounds))])

            #PBO6
            PBO6 = float(self.twobody[line_number][8+2-1]) # 8th term, +2 for atom indices, -1 for 0 indexing
            self.twobody[line_number][8+2-1] = '<<PBO6_'+e1+'_'+e2+'>>'
            self.params_write.append(['PBO6_'+e1+'_'+e2, str(PBO6*(1-bounds)), str(PBO6*(1+bounds))])

            # ro_pipi
            for index, line in enumerate(self.offdiagonal[1:]):
                if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                    break
            line_number = (1 + (1*index))

            # ro_pipi
            ro_pipi = float(self.offdiagonal[line_number][6+2-1]) # 6th term, +2 for atom indices, -1 for 0 indexing
            self.offdiagonal[line_number][6+2-1] = '<<ro_pipi_'+e1+'_'+e2+'>>'
            self.params_write.append(['ro_pipi_'+e1+'_'+e2, str(ro_pipi*(1-bounds)), str(ro_pipi*(1+bounds))])

        return



    def template_bond_energy_attractive(self, e1, e2, double_bond = False, triple_bond = False, bounds = 0.1):
        ie1, ie2 = self.get_element_number(e1), self.get_element_number(e2)

        #-------------------#
        #--- SINGLE BOND ---#
        #-------------------#

        # De_sigma
        for index, line in enumerate(self.twobody[2::2]):
            if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                break
        line_number = (2 + (2*index))

        #De_sigma
        De_sigma = float(self.twobody[line_number][1+2-1]) # 1st term, +2 for atom indices, -1 for 0 indexing
        self.twobody[line_number][1+2-1] = '<<De_sigma_'+e1+'_'+e2+'>>'
        self.params_write.append(['De_sigma_'+e1+'_'+e2, str(De_sigma*(1-bounds)), str(De_sigma*(1+bounds))])

        #PBE1
        PBE1 = float(self.twobody[line_number][4+2-1]) # 4th term, +2 for atom indices, -1 for 0 indexing
        self.twobody[line_number][4+2-1] = '<<PBE1_'+e1+'_'+e2+'>>'
        self.params_write.append(['PBE1_'+e1+'_'+e2, str(PBE1*(1-bounds)), str(PBE1*(1+bounds))])

        line_number = (2 + (2*index) + 1)  # PBE2 is on the next line

        #PBE2
        PBE2 = float(self.twobody[line_number][9-8-1]) # 9th term, -8 for previous line, -1 for 0 indexing
        self.twobody[line_number][9-8-1] = '<<PBE2_'+e1+'_'+e2+'>>'
        self.params_write.append(['PBE2_'+e1+'_'+e2, str(PBE2*(1-bounds)), str(PBE2*(1+bounds))])

        #-------------------#
        #--- DOUBLE BOND ---#
        #-------------------#

        # De_pi
        for index, line in enumerate(self.twobody[2::2]):
            if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                break
        line_number = (2 + (2*index))

        #De_pi
        De_pi = float(self.twobody[line_number][2+2-1]) # 2nd term, +2 for atom indices, -1 for 0 indexing
        self.twobody[line_number][2+2-1] = '<<De_pi_'+e1+'_'+e2+'>>'
        self.params_write.append(['De_pi_'+e1+'_'+e2, str(De_pi*(1-bounds)), str(De_pi*(1+bounds))])

        #-------------------#
        #--- TRIPLE BOND ---#
        #-------------------#

        # De_pipi
        for index, line in enumerate(self.twobody[2::2]):
            if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                break
        line_number = (2 + (2*index))

        #De_pipi
        De_pipi = float(self.twobody[line_number][3+2-1]) # 3rd term, +2 for atom indices, -1 for 0 indexing
        self.twobody[line_number][3+2-1] = '<<De_pipi_'+e1+'_'+e2+'>>'
        self.params_write.append(['De_pipi_'+e1+'_'+e2, str(De_pipi*(1-bounds)), str(De_pipi*(1+bounds))])











    def template_bond_energy_vdW(self, e1, e2, f13 = False, bounds = 0.1):
        ie1, ie2 = self.get_element_number(e1), self.get_element_number(e2)

        # Dij, rvdWm alpha_ij in off-diagonal
        for index, line in enumerate(self.offdiagonal[1:]):
            if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                break
        line_number = (1 + (1*index))

        # Dij
        Dij = float(self.offdiagonal[line_number][1+2-1]) # 1st term, +2 for atom indices, -1 for 0 indexing
        self.offdiagonal[line_number][1+2-1] = '<<Dij_'+e1+'_'+e2+'>>'
        self.params_write.append(['Dij_'+e1+'_'+e2, str(Dij*(1-bounds)), str(Dij*(1+bounds))])

        # rvdW
        rvdW = float(self.offdiagonal[line_number][2+2-1]) # 2nd term, +2 for atom indices, -1 for 0 indexing
        self.offdiagonal[line_number][2+2-1] = '<<rvdW_'+e1+'_'+e2+'>>'
        self.params_write.append(['rvdW_'+e1+'_'+e2, str(rvdW*(1-bounds)), str(rvdW*(1+bounds))])

        # alpha_ij
        alpha_ij = float(self.offdiagonal[line_number][3+2-1]) # 3rd term, +2 for atom indices, -1 for 0 indexing
        self.offdiagonal[line_number][3+2-1] = '<<alpha_ij_'+e1+'_'+e2+'>>'
        self.params_write.append(['alpha_ij_'+e1+'_'+e2, str(alpha_ij*(1-bounds)), str(alpha_ij*(1+bounds))])


        ### WRITE PARAMETER IN F13
        if f13:
            # gamma_w in element 1 in onebody
            for index, line in enumerate(self.onebody[4::4]):
                if line[0].lower().upper() == e1.lower().upper():
                    break
            line_number = (4 + (4*index) + 1)

            # gamma_w
            gamma_w = float(self.onebody[line_number][10-8-1]) # 10th term, -8 for previous line, -1 for 0 indexing
            self.onebody[line_number][3+2-1] = '<<gamma_w_'+e1+'>>'
            self.params_write.append(['gamma_w_'+e1, str(gamma_w*(1-bounds)), str(gamma_w*(1+bounds))])

            # gamma_w in element 2 in onebody
            for index, line in enumerate(self.onebody[4::4]):
                if line[0].lower().upper() == e2.lower().upper():
                    break
            line_number = (4 + (4*index) + 1)

            # gamma_w
            gamma_w = float(self.onebody[line_number][10-8-1]) # 10th term, -8 for previous line, -1 for 0 indexing
            self.onebody[line_number][3+2-1] = '<<gamma_w_'+e2+'>>'
            self.params_write.append(['gamma_w_'+e2, str(gamma_w*(1-bounds)), str(gamma_w*(1+bounds))])


            # Pvdw1 in general parameters
            line_number = 1 + 29  # 29th parameter, +1 for header line
            P_vdW1 = float(self.general[line_number][0])
            self.general[line_number][0] = '<<PvdW>>'
            self.params_write.append(['PvdW', str(P_vdW1*(1-bounds)), str(P_vdW1*(1+bounds))])








    def template_threebody_energy(self, e1, e2, e3, bounds = 0.1):
        # theta0, Pval1, Pval2 in threebody
        for triplet in list(set(list(itertools.permutations([e1,e2,e3])))):
            ie1 = self.get_element_number(triplet[0])
            ie2 = self.get_element_number(triplet[1])
            ie3 = self.get_element_number(triplet[2])
            for index, line in enumerate(self.threebody[1:]):
                if int(line[0]) == ie1 and int(line[1]) == ie2 and int(line[2]) == ie3:
                    line_number = (1 + (1*index))

                    theta0 = float(self.threebody[line_number][1+3-1]) # 1st term, +3 for atom indices, -1 for 0 indexing
                    self.threebody[line_number][1+3-1] = '<<theta0_'+triplet[0]+'_'+triplet[1]+'_'+triplet[2]+'>>'
                    self.params_write.append(['theta0_'+triplet[0]+'_'+triplet[1]+'_'+triplet[2], str(theta0*(1-bounds)), str(theta0*(1+bounds))])

                    Pval1 = float(self.threebody[line_number][2+3-1]) # 2nd term, +3 for atom indices, -1 for 0 indexing
                    self.threebody[line_number][2+3-1] = '<<Pval1_'+triplet[0]+'_'+triplet[1]+'_'+triplet[2]+'>>'
                    self.params_write.append(['Pval1_'+triplet[0]+'_'+triplet[1]+'_'+triplet[2], str(Pval1*(1-bounds)), str(Pval1*(1+bounds))])

                    Pval2 = float(self.threebody[line_number][3+3-1]) # 3rd term, +3 for atom indices, -1 for 0 indexing
                    self.threebody[line_number][3+3-1] = '<<Pval2_'+triplet[0]+'_'+triplet[1]+'_'+triplet[2]+'>>'
                    self.params_write.append(['Pval2_'+triplet[0]+'_'+triplet[1]+'_'+triplet[2], str(Pval2*(1-bounds)), str(Pval2*(1+bounds))])




    def template_fourbody_energy(self, e1, e2, e3, e4, bounds = 0.1):
        # V1, V2, V3, Ptor1 in fourbody
        for quartet in list(set(list(itertools.permutations([e1,e2,e3,e4])))):
            ie1 = self.get_element_number(quartet[0])
            ie2 = self.get_element_number(quartet[1])
            ie3 = self.get_element_number(quartet[2])
            ie4 = self.get_element_number(quartet[3])

            for index, line in enumerate(self.fourbody[1:]):
                if int(line[0]) == ie1 and int(line[1]) == ie2 and int(line[2]) == ie3 and int(line[3]) == ie4:
                    line_number = (1 + (1*index))

                    V1 = float(self.fourbody[line_number][1+4-1]) # 1st term, +3 for atom indices, -1 for 0 indexing
                    self.fourbody[line_number][1+4-1] = '<<V1_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3]+'>>'
                    self.params_write.append(['V1_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3], str(V1*(1-bounds)), str(V1*(1+bounds))])

                    V2 = float(self.fourbody[line_number][2+4-1]) # 2nd term, +3 for atom indices, -1 for 0 indexing
                    self.fourbody[line_number][2+4-1] = '<<V2_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3]+'>>'
                    self.params_write.append(['V2_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3], str(V2*(1-bounds)), str(V2*(1+bounds))])

                    V3 = float(self.fourbody[line_number][3+4-1]) # 3rd term, +3 for atom indices, -1 for 0 indexing
                    self.fourbody[line_number][3+4-1] = '<<V3_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3]+'>>'
                    self.params_write.append(['V3_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3], str(V3*(1-bounds)), str(V3*(1+bounds))])

                    Ptor1 = float(self.fourbody[line_number][4+4-1]) # 4th term, +3 for atom indices, -1 for 0 indexing
                    self.fourbody[line_number][4+4-1] = '<<Ptor1_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3]+'>>'
                    self.params_write.append(['Ptor1_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3], str(Ptor1*(1-bounds)), str(Ptor1*(1+bounds))])



    def generate_templates(self):
        with open(self.ranges, 'w') as ranges_file:
            for parameter in self.params_write:
                ranges_file.write(' '.join(parameter) + '\n')

        with open(self.template,'w') as template:
            template.write(self.header)
            for line in self.general:
                template.write(' '.join(line)+'\n')
            for line in self.onebody:
                template.write(' '.join(line)+'\n')
            for line in self.twobody:
                template.write(' '.join(line)+'\n')
            for line in self.offdiagonal:
                template.write(' '.join(line)+'\n')
            for line in self.threebody:
                template.write(' '.join(line)+'\n')
            for line in self.fourbody:
                template.write(' '.join(line)+'\n')
            for line in self.hbond:
                template.write(' '.join(line)+'\n')




    def make_template_twobody(self, e1, e2, double_bond = False, triple_bond = False, bounds = 0.1, common = False):
        # GET BOND_ORDER_PARAMETERS
        self.template_bond_order(e1,e2,double_bond = double_bond, triple_bond = triple_bond, bounds = bounds)
        self.template_bond_energy_attractive(e1,e2,double_bond = double_bond, triple_bond = triple_bond, bounds = bounds)
        self.template_bond_energy_vdW(e1,e2, f13 = common, bounds = bounds)
        return

    def make_template_threebody(self, e1, e2, e3, bounds = 0.1, common = False):
        self.template_threebody_energy(e1, e2, e3, bounds = bounds)
        return

    def make_template_fourbody(self, e1, e2, e3, e4, bounds = 0.1, common = False):
        self.template_fourbody_energy(e1, e2, e3, e4, bounds = bounds)
        return