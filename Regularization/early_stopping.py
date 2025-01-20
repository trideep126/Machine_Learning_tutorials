def train_with_early_stopping(model,train_data,val_data,epochs,patience):
 best_val_loss = float('inf')
 patience_counter = 0

 for epoch in range(epochs):
    model.fit(train_data[0],train_data[1])
    val_loss = model.evaluate(val_data[0],val_data[1])

    if val_loss < best_val_loss:
        best_val_loss = val_loss 
        patience_counter = 0 
    else:
       patience_counter += 1

    if patience_counter >= patience:
       print(f"Early Stopping at epoch{epoch}")
       break

    return model

