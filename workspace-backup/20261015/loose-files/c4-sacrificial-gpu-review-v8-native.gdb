set pagination off
set confirm off
set breakpoint pending on
set print thread-events off
python
import gdb
class NativeEntry(gdb.Breakpoint):
    def __init__(self, symbol, label):
        super().__init__(symbol, internal=False)
        self.label = label
        self.entries = 0
    def stop(self):
        self.entries += 1
        return False
points = [
    NativeEntry("THPEvent_query(_object*, _object*)", "event_query"),
    NativeEntry("THPEvent_wait(_object*, _object*, _object*)", "event_wait"),
    NativeEntry("THPStream_query(_object*, _object*)", "stream_query"),
    NativeEntry("THPEvent_pynew(_typeobject*, _object*, _object*)", "event_constructor"),
    NativeEntry("THPStream_pynew(_typeobject*, _object*, _object*)", "stream_constructor"),
]
end
run
python
for point in points:
    print("NATIVE_BODY_ENTRIES", point.label, point.entries)
end
