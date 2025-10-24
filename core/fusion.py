from .utils import time_of_day_label, clothing_style_from_colors

def fuse(frame, tracked_persons, face_infos):
    '''
    tracked_persons: [{'id':int, 'box':(x1,y1,x2,y2)}]
    face_infos: aligned list from FaceAttrs.analyze()
    returns:
      context dict for AdEngine
      enriched_persons (with attrs)
    '''
    persons = []
    for p, f in zip(tracked_persons, face_infos):
        x1,y1,x2,y2 = [int(v) for v in p['box']]
        torso = frame[int(y1+0.3*(y2-y1)):int(y1+0.8*(y2-y1)), x1:x2]
        style = clothing_style_from_colors(torso)
        person = {'id': p['id'], 'box': p['box'],
                  'age': f['age'], 'gender': f['gender'],
                  'expression': f['expression'] or 'neutral',
                  'clothing_style': style,
                  'is_child': f['is_child']}
        persons.append(person)

    ctx = {
        'people_count': len(persons),
        'time_of_day': time_of_day_label(),
        'persons': persons
    }
    return ctx, persons
