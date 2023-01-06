
import sys
import code
from collections import defaultdict
PERMISSIVE=False
MONO=True
mono = defaultdict(set)

cunt =  0
def boundaries_from_celex(eml_path, emw_path):
    vocab = {}
    with open(eml_path) as f:
        for line in f:
            chunks = line.rstrip().split('\\')
            surface = chunks[1]
            imm = chunks[11]
            morphstatus = chunks[3]
            trans = chunks[17]
            if ' ' in surface:
                continue
            imm_segments = imm.split('+')
            if len(imm_segments) == 1:
                mono[surface].add(morphstatus)
                continue
            if surface in mono and "Z" in mono[surface]:
                global cunt
                cunt += 1
                print("FAAAAAAAK", cunt)
            surface_clean = ''.join(c for c in surface if c.isalpha())
            clean_to_orig_map = []
            for i, c in enumerate(surface):
                if c.isalpha():
                    clean_to_orig_map.append(i)
            trans_segments = trans.split('#')
            boundaries = []
            segment_posn = 0
            bad_deriv = False
            # print("#B1")
            # code.interact(local=locals())
            for imm_segment, trans_segment in zip(imm_segments, trans_segments):
                imm_segment = ''.join(c for c in imm_segment if c.isalpha())
                imm_sp = imm_segment.find(' ')
                if imm_sp != -1 and trans_segment.startswith('-'):
                    sub_end = trans_segment.find('+')
                    if sub_end == -1:
                        sub_end = len(trans_segment)
                    if trans_segment[1:sub_end].endswith(imm_segment[imm_sp+1:]):
                        cut_len = len(imm_segment[imm_sp+1:])
                        imm_segment = imm_segment[:imm_sp]
                        if sub_end == cut_len + 1:
                            trans_segment = trans_segment[sub_end:]
                        else:
                            trans_segment = trans_segment[:sub_end-cut_len] + trans_segment[sub_end:]
                    else:
                        bad_deriv = True
                        # print("#B5")
                        # code.interact(local=locals())
                        break
                    # print("#B4")
                    # code.interact(local=locals())
                segment_end_posns = []
                if trans_segment.startswith('-'):
                    sub_end = trans_segment.find('+')
                    if sub_end == -1:
                        surf_segment_len = len(imm_segment) - (len(trans_segment) - 1)
                        for i in range(len(trans_segment) - 1):
                            if segment_posn+surf_segment_len+i >= len(surface_clean):
                                i = 0
                                break
                            try:
                            # figure out how many of the elided characters are shared with the surface form
                                if imm_segment[surf_segment_len+i] != surface_clean[segment_posn+surf_segment_len+i]:
                                    break
                            except IndexError as e:
                                print(f'surface:{surface}, surface_clean:{surface_clean}, imm_segments:{imm_segments}, imm_segment:{imm_segment}, trans_segment:{trans_segment}, segment_posn:{segment_posn}, surf_segment_len:{surf_segment_len}, i:{i}')
                                raise e
                        if i > 0:
                            segment_end_posns.append(segment_posn+surf_segment_len+i)
                    else:
                        surf_segment_len = (len(imm_segment) - (sub_end-1)) + len(trans_segment) - (sub_end+1)
                elif trans_segment.startswith('+'):
                    surf_segment_len = len(imm_segment) + len(trans_segment) - 1
                else:
                    surf_segment_len = len(imm_segment)
                segment_posn += surf_segment_len
                segment_end_posns.append(segment_posn)
                boundaries.append(segment_end_posns)
                # print("#B2")
                # code.interact(local=locals())
            if bad_deriv:
                continue
            del boundaries[-1]
            try:
                boundaries = [[clean_to_orig_map[b] for b in b_alts] for b_alts in boundaries]
            except IndexError as e:
                print(f'surface:{surface}, surface_clean:{surface_clean}, imm_segments:{imm_segments}, trans_segments:{trans_segments}, boundaries:{boundaries}, clean_to_orig_map:{clean_to_orig_map}')
                raise e
            vocab[surface] = (imm_segments, boundaries)
            # print("#B3")
            # code.interact(local=locals)
    with open(emw_path) as f:
        for line in f:
            chunks = line.rstrip().split('\\')
            surface = chunks[1]
            trans = chunks[5]
            if ' ' in surface:
                continue
            if chunks[4] == 'S' or (chunks[4] == 'P' and not surface.endswith('s')):
                vocab[surface+"'s"] = ([surface, "s"], [[len(surface)]])
            if (chunks[4] == 'P' or chunks[4] == 'S') and surface.endswith('s'):
                vocab[surface+"'"] = ([surface], [[len(surface)]])
            sub_start = trans.find('-')
            add_start = trans.find('+')
            if add_start == -1:
                continue
            sub = ''
            if sub_start != -1:
                sub = trans[sub_start+1:add_start]
            add = trans[add_start+1:]
            source = surface[:len(surface)-len(add)] + sub
            imm_segments = [source, add]
            boundaries = [[len(surface)-len(add)]]
            vocab[surface] = (imm_segments, boundaries)
    # link segmentations
    vocab_done = {}
    def link_segmentation(surface):
        seg_done = vocab_done.get(surface)
        if seg_done is None:
            seg_imm = vocab.get(surface)
            if seg_imm is None:
                return []
            else:
                imm_segments, boundaries = seg_imm
                boundaries_expanded = []
                last = 0
                for imm_segment, boundary in zip(imm_segments, boundaries+[[len(surface)]]):
                    for sub_boundary in link_segmentation(imm_segment):
                        if all(b_alt+last < boundary[-1] for b_alt in sub_boundary):
                            boundaries_expanded.append([b_alt + last for b_alt in sub_boundary])
                        else:
                            break
                    boundaries_expanded.append(boundary)
                    last = boundary[-1]
                if len(imm_segments) >= len(boundaries)+1:
                    del boundaries_expanded[-1]
                vocab_done[surface] = boundaries_expanded
                return boundaries_expanded
        else:
            return seg_done
    for w in vocab:
        link_segmentation(w)

    #postprocess all entries to add boundary after hyphen
    for surface, boundaries in vocab_done.items():
        i = 0
        while i < len(boundaries):
            try:
                if len(boundaries[i]) == 1 and surface[boundaries[i][0]-1] == '-':
                    boundaries.insert(i, [boundaries[i][0]-1])
                    i += 1
            except IndexError as e:
                print(f'surface:{surface}, boundaries:{boundaries}, i:{i}')
                raise e
            i += 1

    return vocab_done

def get_segmented_strings(boundary_vocab):
    def get_alternates(surface, boundaries, i, last):
        if i == len(boundaries) - 1:
            if PERMISSIVE:
                yield [surface[last:]]
            for b_alt in boundaries[i]:
                yield [surface[b_alt:], surface[last:b_alt]]
        else:
            try:
                if PERMISSIVE:
                    for alt in get_alternates(surface, boundaries, i + 1, last):
                        yield alt
                for b_alt in boundaries[i]:
                    for alt in get_alternates(surface, boundaries, i+1, b_alt):
                        alt.append(surface[last:b_alt])
                        yield alt
            except IndexError as e:
                print(f'surface:{surface}, boundaries:{boundaries}, i:{i}, last:{last}')
                raise e
    for surface, boundaries in boundary_vocab.items():
        for alternate in get_alternates(surface, boundaries, 0, 0):
            alternate.reverse()
            yield (surface, alternate)

def extract_mono(eml_path, emw_path):
    vocab = []
    with open(eml_path) as f:
        for line in f:
            chunks = line.rstrip().split('\\')
            surface = chunks[1]
            imm = chunks[11]
            morphstatus = chunks[3]
            trans = chunks[17]
            if ' ' in surface:
                continue
            imm_segments = imm.split('+')
            if len(imm_segments) == 1:
                vocab.append(surface)
    return [(w.lower(), [w.lower()]) for w in vocab]

if __name__ == "__main__":
    if not MONO:
        v = boundaries_from_celex(sys.argv[1]+'/english/eml/eml.cd', sys.argv[1]+'/english/emw/emw.cd')
        pairs = get_segmented_strings(v)
    else:
        pairs = extract_mono(sys.argv[1]+'/english/eml/eml.cd', sys.argv[1]+'/english/emw/emw.cd')
    with open(sys.argv[2], mode='w') as out_f:
        for surface, segmentation in pairs:
            out_f.write(f'{surface}\t{" ".join(segmentation)}\n')




