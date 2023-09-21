import numpy as np
import serval_plotting
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import serval_help
import numpy as np
import matplotlib.pyplot as plt
from filepath import FilePath
import pandas as pd

def print_mask(xmin,xmax,delta=0.036):
    print("#########################")
    print("{:0.3f}       0.0000000".format(xmin - delta))
    print("{:0.3f}       1.0000000".format(xmin))
    print("{:0.3f}       1.0000000".format(xmax))
    print("{:0.3f}       0.0000000".format(xmax + delta))
    print("#########################")


def print_min(df,nummin=3):
    """
    Function to print the top minima for a given selection
    """
    print("#########################")
    df = df.sort_values('ff')
    for i in range(nummin):
        print("min#{} w={:10.3f} f={:10.3f}".format(i,df.ww.values[i],df.ff.values[i]))
    print("#########################")


def mask_selector_print_min(ww,ff,w_mask=None,f_mask=None,lines=None,
        nummin=3):
    """
    Use a span selector to print the top minima within the selection.
    Useful to find the minimum point of a stellar line
    
    INPUT:
        ww 
        ff
        
    NOTES:
        Be sure to assign it as a variable, otherwise it gets garbage collected
        s = mask_selector(w,f)
    """

    def onselect(xmin,xmax,nummin=10):
        df = pd.DataFrame(zip(ww,ff),columns=['ww','ff'],index=ww)
        mask = (ww > xmin) & (ww < xmax)
        df = df[mask]
        df = df.sort_values('ff')
        print_min(df,nummin)
        
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(ww, ff, '-')
    ax.set_title('Press left mouse button and drag to test')
        
    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'))
    ax.grid(lw=0.5,alpha=0.5)
    ax.set_xlim(ww[0],ww[-1])

    ylim = ax.get_ylim()
    
    if w_mask is not None:
        ax.fill_between(w_mask,f_mask*ylim[1],alpha=0.2)
    if lines is not None:
        ax.vlines(lines,ymin=ylim[0],ymax=ylim[1],color='orange')
        
    plt.show()
    return span

def mask_selector(w,f):
    """
    Use a span selector to print masks. Useful for telluric masking.
    
    INPUT:
        w: 
        f:
        
    NOTES:
        Be sure to assign it as a variable, otherwise it gets garbage collected
        s = mask_selector(w,f)
    """
    def onselect(xmin, xmax):
        print_mask(xmin,xmax)
        
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(w, f, '-')
    ax.set_title('Press left mouse button and drag to test')
        
    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'))
    plt.show()
    return span


def loop_plot_order(splist,savefolder='loop_select/',plot_fiber='sci sky cal',plot_raw=False,ylim=None):
    """
    Interactive fits image plotting, loops through images in a list using left and right arrow keys.
    
    INPUT:
    
    NOTES:
    - press left to go back
    - press right arrow to move forward
    - press up arrow to print filename
    
    EXAMPLE:
        loop_plot_orders(fitsfiles)
    """
    fitsfiles = [sp.S.filename for sp in splist]
    global ORDER
    global curr_pos
    global object_name
    curr_pos = 0
    ORDER = 18
    utils.make_dir(savefolder)
    selected_frames=[]
    def key_event(e):
        global curr_pos
        global ORDER
        global object_name
        if e.key == "right": curr_pos += 1
        elif e.key == "left": curr_pos -= 1
        elif e.key == "up": ORDER += 1
        elif e.key == "down": ORDER -= 1
        elif e.key == 'enter': 
            print(fitsfiles[curr_pos])
            selected_frames.append(fitsfiles[curr_pos])
        elif e.key == 'shift':
            print(selected_frames)
        elif e.key == ' ':
            # Selected
            filename = savefolder + object_name + "_selected.csv"
            df = pd.DataFrame(zip(selected_frames),columns=['frames'])
            df.to_csv(filename,index=False)
            print("Saved {} selected frames to {}".format(len(selected_frames),filename))

            # Not Selected
            filename = savefolder + object_name + "_notselected.csv"
            notselected_frames = utils.remove_items_from_list(fitsfiles,selected_frames)
            df = pd.DataFrame(zip(notselected_frames),columns=['frames'])
            df.to_csv(filename,index=False)
            print("Saved {} unselected frames to {}".format(len(notselected_frames),filename))
        else: 
            return
        curr_pos = curr_pos % len(splist)
        ax.cla()
        sp = splist[curr_pos]
        object_name = sp.S.obj
        if 'sci' in plot_fiber:
            if plot_raw:
                ax.plot(sp.S._f_sci[ORDER]/sp.S.exptime,color='blue',label='sci',ls='-')
            else:
                ax.plot(sp.S._f_sci[ORDER],color='blue',label='sci',ls='-')
            #ax.plot(sp.S.f_sci[ORDER],color='indigo',label='sci')
        if 'sky' in plot_fiber:
            if plot_raw:
                ax.plot(sp.S._f_sky[ORDER]/sp.S.exptime,color='forestgreen',label='sky')
            else:
                ax.plot(sp.S._f_sky[ORDER],color='forestgreen',label='sky')
            #ax.plot(sp.S.f_sky[ORDER],color='forestgreen',label='sky')
        if 'cal' in plot_fiber:
            if plot_raw:
                ax.plot(sp.S._f_cal[ORDER]/sp.S.exptime,color='orange',label='cal')
            else:
                ax.plot(sp.S._f_cal[ORDER],color='orange',label='cal')
        ax.legend(loc='upper left')
        sn = np.nanmedian(sp.S._f_sci[ORDER]/sp.S.e[ORDER])
        ax.set_title('{} Order={} \n#{}/{}: {}, SNR={:0.1f}'.format(object_name,ORDER,curr_pos+1,len(splist),
                                                        FilePath(fitsfiles[curr_pos]).basename,sn))
        ax.set_xlabel('pixel')
        ax.set_ylabel('counts')
        if ylim is not None:
            ax.set_ylim(*ylim)
        fig.canvas.draw_idle()
        
    # Plot
    fig = plt.figure(figsize=(12,6))
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = fig.add_subplot(111)
    plt.show()

def loop_plot_stacked_orders(splist,savefolder='loop_select/',orders=[3,4,5,6,7,14,15,16,17,18],plot_fibers='sci sky cal'):
    """
    Interactive fits image plotting, loops through images in a list using left and right arrow keys.
    
    INPUT:
    
    NOTES:
    - press left to go back
    - press right arrow to move forward
    - press up arrow to print filename
    
    EXAMPLE:
        loop_plot_orders(fitsfiles)
    """
    fitsfiles = [sp.S.filename for sp in splist]
    global SEP
    global curr_pos
    global object_name
    curr_pos = 0
    SEP = 0.05
    utils.make_dir(savefolder)
    selected_frames=[]
    def key_event(e):
        global curr_pos
        global SEP
        global object_name
        if e.key == "right": curr_pos += 1
        elif e.key == "left": curr_pos -= 1
        elif e.key == "up": SEP += 0.05
        elif e.key == "down": SEP -= 0.05
        elif e.key == 'enter': 
            print(fitsfiles[curr_pos])
            selected_frames.append(fitsfiles[curr_pos])
        elif e.key == 'shift':
            print(selected_frames)
        elif e.key == ' ':
            # Selected
            filename = savefolder + object_name + "_selected.csv"
            df = pd.DataFrame(zip(selected_frames),columns=['frames'])
            df.to_csv(filename,index=False)
            print("Saved {} selected frames to {}".format(len(selected_frames),filename))

            # Not Selected
            filename = savefolder + object_name + "_notselected.csv"
            notselected_frames = utils.remove_items_from_list(fitsfiles,selected_frames)
            df = pd.DataFrame(zip(notselected_frames),columns=['frames'])
            df.to_csv(filename,index=False)
            print("Saved {} unselected frames to {}".format(len(notselected_frames),filename))
        else: 
            return
        curr_pos = curr_pos % len(splist)
        ax.cla()
        sp = splist[curr_pos]
        object_name = sp.S.obj
        for i in orders:#range(spt.S.f.shape[0]):
            if 'sci' in plot_fibers:
                ax.plot(i*SEP + sp.S._f_sci[i],lw=1)
            if 'sky' in plot_fibers:
                ax.plot(i*SEP + sp.S._f_sky[i],lw=2)
            if 'cal' in plot_fibers:
                ax.plot(i*SEP + sp.S._f_cal[i],lw=3)

        sn18 = np.nanmedian(sp.S._f_sci[18]/sp.S.e[18])
        ax.set_title('{} SEP={} Orders={} \n#{}/{}: {}, SNR={:0.1f}'.format(object_name,SEP,orders,curr_pos+1,len(splist),
                                                        FilePath(fitsfiles[curr_pos]).basename,sn18))
        ax.set_xlabel('pixel')
        ax.set_ylabel('counts')
        fig.canvas.draw_idle()
        
    # Plot
    fig = plt.figure(figsize=(12,6))
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = fig.add_subplot(111)
    plt.show()
    
def loop_plot_orders(splist,savefolder='loop_select/',filetype="png"):
    """
    Interactive fits image plotting, loops through images in a list using left and right arrow keys.
    
    INPUT:
    
    NOTES:
    - press left to go back
    - press right arrow to move forward
    - press up arrow to print filename
    
    EXAMPLE:
        loop_plot_orders(fitsfiles)
    """
    fitsfiles = [sp.S.filename for sp in splist]
    global ORDER
    global curr_pos
    global object_name
    curr_pos = 0
    ORDER = 5
    utils.make_dir(savefolder)
    selected_frames=[]
    def key_event(e):
        global curr_pos
        global ORDER
        global object_name
        if e.key == "right": curr_pos += 1
        elif e.key == "left": curr_pos -= 1
        elif e.key == "up": ORDER += 1
        elif e.key == "down": ORDER -= 1
        elif e.key == 'enter': 
            print(fitsfiles[curr_pos])
            selected_frames.append(fitsfiles[curr_pos])
        elif e.key == 'shift':
            print(selected_frames)
        elif e.key == ' ':
            #filename = savefolder + FilePath(fitsfiles[curr_pos]).basename + "." + filetype
            #e.canvas.figure.savefig(filename,dpi=200)
            
            # Selected
            filename = savefolder + object_name + "_selected.csv"
            df = pd.DataFrame(zip(selected_frames),columns=['frames'])
            df.to_csv(filename,index=False)
            print("Saved {} selected frames to {}".format(len(selected_frames),filename))

            # Not Selected
            filename = savefolder + object_name + "_notselected.csv"
            notselected_frames = utils.remove_items_from_list(fitsfiles,selected_frames)
            df = pd.DataFrame(zip(notselected_frames),columns=['frames'])
            df.to_csv(filename,index=False)
            print("Saved {} unselected frames to {}".format(len(notselected_frames),filename))
        else: 
            return
        curr_pos = curr_pos % len(splist)
        ax.cla()
        bx.cla()
        sp = splist[curr_pos]
        object_name = sp.S.obj
        #sp = sspectrum.SSpectrum(fitsfiles[curr_pos],targetname='GJ699',inst="HPF",verbose=False)
        sp.plot_orders(o=ORDER,ax=ax,bx=bx)
        sp.ax.set_title('{} Order={} #{}/{}: {}, SNR={:0.1f}'.format(object_name,ORDER,curr_pos+1,len(splist),
                                                        FilePath(fitsfiles[curr_pos]).basename,sp.S.sn18))
        fig.canvas.draw_idle()
        
    # Plot
    fig = plt.figure(figsize=(12,6))
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = fig.add_subplot(111)
    bx = ax.twinx()
    plt.show()
    
def loop_serval_prervs(splist,spt):
    """
    Interactive fits image plotting, loops through images in a list using left and right arrow keys.
    
    INPUT:
    
    NOTES:
    - press left to go back
    - press right arrow to move forward
    - press up arrow to print filename
    
    EXAMPLE:
    """
    fitsfiles = [sp.S.filename for sp in splist]
    global ORDER
    global curr_pos
    curr_pos = 0
    ORDER = 5
    def key_event(e):
        global curr_pos
        global ORDER
        if e.key == "right": curr_pos += 1
        elif e.key == "left": curr_pos -= 1
        elif e.key == "up": ORDER += 1
        elif e.key == "down": ORDER -= 1
        elif e.key == 'enter': print(fitsfiles[curr_pos])
        else: 
            return
        curr_pos = curr_pos % len(splist)
        ax[0].cla()
        ax[1].cla()
        bx.cla()
        cx.cla()
        sp = splist[curr_pos] 
        _ = serval_help.calculate_pre_rv_for_order(splist[curr_pos],spt,o=ORDER,ax=ax,bx=bx,cx=cx)
        ax[1].set_title('Filename Order={} #{}/{}: {}'.format(ORDER,curr_pos+1,len(splist),
                                                        FilePath(fitsfiles[curr_pos]).basename),y=1.05)
        fig.canvas.draw_idle()
        
    # Plot
    fig = plt.figure(figsize=(30,15))
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = [fig.add_subplot(121),fig.add_subplot(122)]
    #fig, ax = plt.subplots(ncols=2,figsize=(15,7))
    #serval_help.calculate_pre_rv_for_order(splist[0],splist[10],o=5,ax=ax)
    bx = ax[1].twinx()
    cx = ax[1].twiny()
    plt.show()


def loop_serval_templatervs(splist,spt,order=5,plot_in_pixelspace=False):
    """
    Interactive fits image plotting, loops through images in a list using left and right arrow keys.
    
    INPUT:
    
    NOTES:
    - press left to go back
    - press right arrow to move forward
    - press up arrow to print filename
    
    EXAMPLE:
    """
    fitsfiles = [sp.S.filename for sp in splist]
    global ORDER
    global curr_pos
    curr_pos = 0
    ORDER = order
    def key_event(e):
        global curr_pos
        global ORDER
        if e.key == "right": curr_pos += 1
        elif e.key == "left": curr_pos -= 1
        elif e.key == "up": ORDER += 1
        elif e.key == "down": ORDER -= 1
        elif e.key == 'enter': print(fitsfiles[curr_pos])
        else: 
            return
        curr_pos = curr_pos % len(splist)
        ax[0].cla()
        ax[1].cla()
        bx.cla()
        sp = splist[curr_pos] 
        _ = serval_help.calculate_rv_for_order_from_final_template(splist[curr_pos],spt,o=ORDER,ax=ax,bx=bx,plot_in_pixelspace=plot_in_pixelspace)
        ax[1].set_title('Filename Order={} #{}/{}: {}'.format(ORDER,curr_pos+1,len(splist),
                                                        FilePath(fitsfiles[curr_pos]).basename),y=1.05)
        fig.canvas.draw_idle()
        
    # Plot
    fig = plt.figure(figsize=(30,15))
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = [fig.add_subplot(121),fig.add_subplot(122)]
    #fig, ax = plt.subplots(ncols=2,figsize=(15,7))
    #serval_help.calculate_pre_rv_for_order(splist[0],splist[10],o=5,ax=ax)
    bx = ax[1].twinx()
    plt.show()


def loop_serval_templatervs(splist,spt,order=5,plot_in_pixelspace=False,figsize=(15,7)):
    """
    Interactive fits image plotting, loops through images in a list using left and right arrow keys.
    
    INPUT:
    
    NOTES:
    - press left to go back
    - press right arrow to move forward
    - press up arrow to print filename
    
    EXAMPLE:
    """
    fitsfiles = [sp.S.filename for sp in splist]
    global ORDER
    global curr_pos
    curr_pos = 0
    ORDER = order
    def key_event(e):
        global curr_pos
        global ORDER
        if e.key == "right": curr_pos += 1
        elif e.key == "left": curr_pos -= 1
        elif e.key == "up": ORDER += 1
        elif e.key == "down": ORDER -= 1
        elif e.key == 'enter': print(fitsfiles[curr_pos])
        else: 
            return
        curr_pos = curr_pos % len(splist)
        ax[0].cla()
        ax[1].cla()
        bx.cla()
        sp = splist[curr_pos] 
        _ = serval_help.calculate_rv_for_order_from_final_template(splist[curr_pos],spt,o=ORDER,ax=ax,bx=bx,plot_in_pixelspace=plot_in_pixelspace)
        ax[1].set_title('Filename Order={} #{}/{}: {}'.format(ORDER,curr_pos+1,len(splist),
                                                        FilePath(fitsfiles[curr_pos]).basename),y=1.05)
        fig.canvas.draw_idle()
        
    # Plot
    fig = plt.figure(figsize=figsize)
    fig.canvas.mpl_connect('key_press_event', key_event)
    #ax = [fig.add_subplot(121),fig.add_subplot(122)]
    ax = [fig.add_subplot(211),fig.add_subplot(212)]
    #fig, ax = plt.subplots(nrows=2,figsize=(15,7))
    #serval_help.calculate_pre_rv_for_order(splist[0],splist[10],o=5,ax=ax)
    bx = ax[1].twinx()
    plt.show()

def loop_templatediv(hf,order=5,i=0,rv=0.,airwave=True):
    """
    Interactive fits image plotting, loops through images in a list using left and right arrow keys.
    
    INPUT:
    
    NOTES:
    - press left to go back
    - press right arrow to move forward
    - press up arrow to print filename
    
    EXAMPLE:
    """
    LEN = len(hf['template/tstack_w'])
    global ORDER
    global curr_pos
    curr_pos = 0
    ORDER = order
    def key_event(e):
        global curr_pos
        global ORDER
        if e.key == "right": curr_pos += 1
        elif e.key == "left": curr_pos -= 1
        elif e.key == "up": ORDER += 1
        elif e.key == "down": ORDER -= 1
        else: 
            return
        curr_pos = curr_pos % LEN
        ax.cla()
        serval_plotting.plot_epoch_div_template(hf,ORDER,curr_pos,ax=ax,rv=rv,airwave=airwave)
        #ax[1].set_title('Filename Order={} #{}/{}: {}'.format(ORDER,curr_pos+1,len(splist),
        #                                                FilePath(fitsfiles[curr_pos]).basename),y=1.05)
        fig.canvas.draw_idle()
        
    # Plot
    fig = plt.figure(figsize=(30,15))
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = fig.add_subplot(111)
    #fig, ax = plt.subplots(ncols=2,figsize=(15,7))
    #serval_help.calculate_pre_rv_for_order(splist[0],splist[10],o=5,ax=ax)
    plt.show()


def loop_templatediv2(hf,order=5,i=0,rv=0.,airwave=True):
    """
    Interactive fits image plotting, loops through images in a list using left and right arrow keys.
    
    INPUT:
    
    NOTES:
    - press left to go back
    - press right arrow to move forward
    - press up arrow to print filename
    
    EXAMPLE:
    """
    LEN = hf['template/tstack_w'].shape[1]
    print(LEN)
    global ORDER
    global curr_pos
    curr_pos = 0
    ORDER = order
    def key_event(e):
        global curr_pos
        global ORDER
        if e.key == "right": curr_pos += 1
        elif e.key == "left": curr_pos -= 1
        elif e.key == "up": ORDER += 1
        elif e.key == "down": ORDER -= 1
        else: 
            return
        curr_pos = curr_pos % LEN
        #ax.cla()
        #serval_plotting.plot_epoch_div_template(hf,ORDER,curr_pos,ax=ax,rv=rv,airwave=airwave)
        xx, yy, filename = serval_plotting.plot_epoch_div_template(hf,ORDER,curr_pos,ax=None,rv=rv,airwave=airwave,plot=False)
        line.set_data(xx,yy)
        title = '{}/{} {} exptime={:3.2f} sn19={:3.2f}'.format(curr_pos,LEN,filename,hf['rv/exptime'][:][curr_pos],hf['rv/sn18'][:][curr_pos])
        ax.set_title(title)
        plt.draw()
        #fig.canvas.draw_idle()
        
    # Plot
    fig = plt.figure(figsize=(30,15))
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = fig.add_subplot(111)

    xx, yy, filename = serval_plotting.plot_epoch_div_template(hf,ORDER,curr_pos,ax=None,rv=rv,airwave=airwave,plot=False)
    colors = utils.get_cmap_colors('jet',N=LEN)
    for i in range(LEN):
        _ = serval_plotting.plot_epoch_div_template(hf,ORDER,i,ax=ax,rv=rv,airwave=airwave,plot=True,color=colors[i])
    line = ax.plot(xx,yy,alpha=1,marker='.',color='black')[0]
    ax.set_title(filename)
    plt.show()
