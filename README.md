# dvclive
DVC metrics logger

Usage example:
```python
import dvclive
from dvclive.keras import DvcLiveCallback

dvclive.init('dvclogs')

...
`
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    callbacks=[DvcLiveCallback()])

```
