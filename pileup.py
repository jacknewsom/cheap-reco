import numpy as np

def shift_and_sum_events(events):
    shifted_events = []
    for event in events:
        shift = np.random.random_integers(-95, 95)
        shifted_event = event + np.full(event.shape, shift)
        shifted_events.append(shifted_event)
    return shifted_events
